import os.path as osp

import mmengine.fileio as fileio

from flabplatform.flabseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class A0belt_imagesDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for A0belt_images.
    """
    METAINFO = dict(
        classes=('background', 'person_nobelt', 'Lift_tool',
                 'Union', 'person_belt', 'unclear'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
