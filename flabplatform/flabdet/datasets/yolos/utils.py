import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from flabplatform.flabdet.utils.yolos import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    YOLO_ROOT,
    SETTINGS_FILE,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    yaml_load,
    yaml_save,
)
from flabplatform.flabdet.utils.yolos.checks import check_file, check_font, is_ascii

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"

def img2label_paths(img_paths):
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def img2label_paths_labelme(img_paths):
    sa, sb = f"{os.sep}image{os.sep}", f"{os.sep}label{os.sep}"  # /image/, /label/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".json" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def verify_image(args):
    """Verify one image."""
    (im_file, cls), prefix = args
    # Number (found, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args):
    """Verify one image-label pair."""
    from ultralytics.utils.ops import segments2boxes
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls, name2id = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                if single_cls:
                    lb[:, 0] = 0
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def verify_image_label_labelme(args):
    """Verify one image-label pair."""
    from ultralytics.utils.ops import segments2boxes
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls, name2id = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        
        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f: #TODO 增加对分类的支持，增加对brush的支持
                lb_json = json.load(f)
                image_width = lb_json["imageWidth"]
                image_height = lb_json["imageHeight"]
                lb = []
                for sap in lb_json["shapes"]:
                    if sap["shape_type"] == "rectangle":
                        points = sap["points"]
                        x_min = min(points[0][0], points[1][0])
                        y_min = min(points[0][1], points[1][1])
                        x_max = max(points[0][0], points[1][0])
                        y_max = max(points[0][1], points[1][1])

                        x_center = (x_min + x_max) / 2.0 / image_width
                        y_center = (y_min + y_max) / 2.0 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height
                        class_id = name2id[sap["label"]]
                        lb.append([class_id, x_center, y_center, width, height])
                    else:
                        temp_points = []
                        temp_points.append(name2id[sap["label"]])
                        for point in sap["points"]:
                            x = point[0] / image_width
                            y = point[1] / image_height
                            temp_points.append(x)
                            temp_points.append(y)
                        lb.append(temp_points)
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                if single_cls:
                    lb[:, 0] = 0
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]




def visualize_image_annotations(image_path, txt_path, label_map):
    """
    Visualizes YOLO annotations (bounding boxes and class labels) on an image.

    This function reads an image and its corresponding annotation file in YOLO format, then
    draws bounding boxes around detected objects and labels them with their respective class names.
    The bounding box colors are assigned based on the class ID, and the text color is dynamically
    adjusted for readability, depending on the background color's luminance.

    Args:
        image_path (str): The path to the image file to annotate, and it can be in formats supported by PIL.
        txt_path (str): The path to the annotation file in YOLO format, that should contain one line per object.
        label_map (dict): A dictionary that maps class IDs (integers) to class labels (strings).

    Examples:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # It should include all annotated classes details
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    fig, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True))  # Get and normalize the RGB color
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found.

    Returns:
        (dict): Parsed dataset information and paths.
    """
    from ultralytics.utils.downloads import safe_download
    if isinstance(dataset, str):

        file = check_file(dataset)

        # Download (optional)
        extract_dir = ""
        if zipfile.is_zipfile(file) or is_tarfile(file):
            new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
            file = find_dataset_yaml(DATASETS_DIR / new_dir)
            extract_dir, autodownload = file.parent, False

        # Read YAML
        data = yaml_load(file, append_filename=True)  # dictionary
    elif isinstance(dataset, dict):
        extract_dir = ""
        data = dataset  # already a dictionary

    # Checks
    # for k in "train", "val":
    #     if k not in data:
    #         if k != "val" or "validation" not in data:
    #             raise SyntaxError(
    #                 emojis(f"'{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
    #             )
    #         LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
    #         data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])
    from ultralytics.nn.autobackend import check_class_names
    data["names"] = check_class_names(data["names"])

    # Resolve paths
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            m = f"\nimages not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            raise FileNotFoundError(m)
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts
    return data  # dictionary


def check_cls_dataset(dataset, split=""):
    """
    Checks a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''.

    Returns:
        (dict): A dictionary containing the following keys:
            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'test' (Path): The directory path containing the test set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict): A dictionary of class names in the dataset.
    """
    nc = len(dataset['names'])  # number of classes
    names = dataset['names']  # class names list
    names = dict(enumerate(sorted(names)))
    dataset['names'] = names

    path = Path(dataset.get("path"))
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()

    for k in "train", "val", "test":
        if dataset.get(k):  # prepend path
            if isinstance(dataset[k], str):
                x = (path / dataset[k]).resolve()
                if not x.exists() and dataset[k].startswith("../"):
                    x = (path / dataset[k][3:]).resolve()
                dataset[k] = str(x)
            else:
                dataset[k] = [str((path / x).resolve()) for x in dataset[k]]
    dataset['nc'] = nc
    return dataset


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image.
        quality (int, optional): The image compression quality as a percentage.

    Examples:
        >>> from pathlib import Path
        >>> from ultralytics.data.utils import compress_one_image
        >>> for f in Path("path/to/dataset").rglob("*.jpg"):
        >>>    compress_one_image(f)
    """
    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path, optional): Path to images directory.
        weights (list | tuple, optional): Train, validation, and test split fractions.
        annotated_only (bool, optional): If True, only images with an associated txt file are used.

    Examples:
        >>> from ultralytics.data.utils import autosplit
        >>> autosplit()
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


# 理解这块的优化加速技巧 TODO
def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        with open(str(path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")
