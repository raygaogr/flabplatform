import copy
import os.path as osp
from typing import List, Union

from flabplatform.flabdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset
from pathlib import Path
import glob
import os
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"} 
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}"

@DATASETS.register_module()
class LabelmeDetDataset(BaseDetDataset):
    """Dataset for Labelme."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        im_files = self.get_img_files()
        coco_format = self.labelme_to_coco(im_files, self.metainfo['classes'])
        
        self.coco = self.COCOAPI(dataset=coco_format)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
    
    def labelme_to_coco(self, img_files: List[str],
                       class_names: List[str] = None) -> dict:
        """
        Convert Labelme format dataset to COCO format.
        
        Args:
            img_files (List[str]): List of image files to convert
            class_names (List[str], optional): Predefined class names
        """
        import json
        import datetime
        
        # Initialize COCO format
        coco_format = {
            "info": {
                "description": "Labelme to COCO converted dataset",
                "url": "",
                "version": "1.0", 
                "year": datetime.datetime.now().year,
                "contributor": "Labelme2COCO Converter",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories to COCO format
        category_name_to_id = {}
        for i, class_name in enumerate(class_names):
            coco_format["categories"].append({
                "id": i + 1,  # COCO categories start from 1
                "name": class_name,
                "supercategory": "none"
            })
            category_name_to_id[class_name] = i + 1
        
        # Process each labelme file
        image_id = 0
        annotation_id = 0
        processed_count = 0
        
        for img_file in img_files:
            try:
                json_name = osp.splitext(osp.basename(img_file))[0] + '.json'
                json_file = osp.join(osp.dirname(osp.dirname(img_file)), "label", json_name) 
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                width = data.get('imageWidth', 0)
                height = data.get('imageHeight', 0)

                # Add image info
                image_id += 1
                image_info = {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": osp.basename(img_file),
                    "license": 1,
                    "date_captured": ""
                }
                coco_format["images"].append(image_info)
                
                # Process annotations
                for shape in data.get('shapes', []):
                    label = shape.get('label', '').strip()
                    if not label or label not in category_name_to_id:
                        continue
                    
                    points = shape.get('points', [])
                    shape_type = shape.get("shape_type", "")
                    
                    # For detection, we only handle rectangles
                    if len(points) != 2 or shape_type != "rectangle": 
                        continue
                    
                    # Calculate bounding box from rectangle points
                    x_coords = [point[0] for point in points]
                    y_coords = [point[1] for point in points]
                    
                    x_min = max(0, min(x_coords))
                    y_min = max(0, min(y_coords))
                    x_max = min(width, max(x_coords))
                    y_max = min(height, max(y_coords))
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue
                    
                    # Calculate area for detection box (simple rectangle area)
                    area = bbox_width * bbox_height
                    
                    # Create detection annotation (no segmentation for detection)
                    annotation_id += 1
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_name_to_id[label],
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "segmentation": [],  # No segmentation for detection
                        "area": area,
                        "iscrowd": 0
                    }
                    
                    coco_format["annotations"].append(annotation)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{len(img_files)} files")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        return coco_format
    
    def get_img_files(self):
        """
        Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        # img_path = [osp.join(self.data_root, x) for x in self.data_prefix["img"]]
        img_path = self.data_prefix["img"]
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    if str(p).split(".")[-1].lower() in IMG_FORMATS:
                        f.append(str(p))
                    else:
                        with open(p, encoding="utf-8") as t:
                            t = t.read().strip().splitlines()
                            parent = str(p.parent) + os.sep
                            f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                            # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n") from e
        return im_files


    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        if Path(self.data_prefix["img"][0]).is_dir():
            img_path = osp.join(self.data_prefix["img"][0], "image", img_info['file_name'])
        else:
            img_path = osp.join(osp.dirname(self.data_prefix["img"][0]), img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
