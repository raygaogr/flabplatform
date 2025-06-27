import os
import os.path
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.vision import VisionDataset
import json
from flabplatform.flabdet.utils import LOGGER


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_instance(img_file_path: str, class_to_idx: dict) -> Tuple[str, int]:
    label_file_basename = os.path.basename(img_file_path).strip().split('.')[0] + ".json"
    img_dir = str(Path(os.path.dirname(img_file_path)).parent) 
    label_file_path = os.path.join(img_dir, "label", label_file_basename)
    with open(label_file_path, 'r') as f:
        label_dict = json.load(f)
        if len(label_dict['shapes']) != 1:
            raise ValueError(f"Expected exactly one classification label, but found {len(label_dict['shapes'])} in {label_file_path}.")
        if label_dict['shapes'][0]['shape_type'] != 'classification':
            raise ValueError(f"Only classification labels are supported, but found a {label_dict['shapes'][0]['shape_type']} shape type.")
        target_class = label_dict['shapes'][0]['label']
        target_class_idx = class_to_idx.get(target_class, None)
        if target_class_idx is None:
            raise ValueError(f"Class '{target_class}' not found in class_to_idx mapping.")
    return (img_file_path, target_class_idx), target_class
            



def make_dataset(
    directories: List[Union[str, Path]],
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    """
    # directory = os.path.expanduser(directory)

    # if class_to_idx is None:
    #     _, class_to_idx = find_classes(directory)
    # elif not class_to_idx:
    #     raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_item in directories:
        if os.path.isfile(target_item) and is_valid_file(target_item):
            item, class_name = get_instance(target_item, class_to_idx)
            instances.append(item)
            available_classes.add(class_name)
        elif os.path.isdir(target_item) and os.path.exists(target_item):
            for root, _, fnames in sorted(os.walk(os.path.join(target_item, 'image'), followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item, class_name = get_instance(path, class_to_idx)
                        instances.append(item)
                        available_classes.add(class_name)
        else:
            LOGGER.info(f"Directory '{target_item}' does not exist or is not a valid file.")

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        data: Any = None
    ) -> None:
        if os.path.isdir(root[0]):
            cache_root = root[0]
        elif os.path.isfile(root[0]):
            cache_root = os.path.dirname(root[0])
        super().__init__(cache_root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(data)
        samples = self.make_dataset(
            root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        """
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
        )

    def find_classes(self, data: Dict[str, Dict[int, str]]) -> Tuple[List[str], Dict[str, int]]:

        return list(data["names"].values()), dict(zip(data["names"].values(), data["names"].keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        data: Any = None
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            data=data
        )
        self.imgs = self.samples
