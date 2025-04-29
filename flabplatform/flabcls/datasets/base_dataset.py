import os.path as osp
from os import PathLike
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset as _BaseDataset

from flabplatform.flabcls.registry import DATASETS, TRANSFORMS


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


@DATASETS.register_module()
class BaseDataset(_BaseDataset):
    """图像分类任务的基础数据集类。

    参数:
        ann_file (str): 标注文件路径。
        metainfo (dict, optional): 数据集的元信息，例如类别信息。默认值为 None。
        data_root (str): ``data_prefix`` 和 ``ann_file`` 的根目录。默认值为 ''。
        data_prefix (str | dict): 训练数据的前缀。默认值为 ''。
        filter_cfg (dict, optional): 数据过滤的配置。默认值为 None。
        indices (int 或 Sequence[int], optional): 支持使用标注文件中的前几个数据，
            以便在较小的数据集上进行训练/测试。默认值为 None，表示使用所有的 ``data_infos``。
        serialize_data (bool): 是否使用序列化对象来保存内存。
            启用后，数据加载器的工作线程可以使用主进程的共享内存，而不是创建副本。默认值为 True。
        pipeline (Sequence): 数据处理流水线。默认值为空元组。
        test_mode (bool, optional): ``test_mode=True`` 表示处于测试阶段，
            如果获取某个数据项失败，将引发错误；
            ``test_mode=False`` 表示处于训练阶段，将随机返回另一个数据项。默认值为 False。
        lazy_init (bool): 是否在实例化时加载标注。
            在某些情况下，例如可视化，仅需要数据集的元信息，而不需要加载标注文件。
            通过设置 ``lazy_init=False``，可以跳过加载标注以节省时间。默认值为 False。
        max_refetch (int): 如果 ``BaseDataset.prepare_data`` 获取到 None 图像，
            尝试获取有效图像的最大额外循环次数。默认值为 1000。
        classes (str | Sequence[str], optional): 指定类别名称。

            - 如果是字符串，则应为文件路径，文件中的每一行是一个类别名称。
            - 如果是字符串序列，则每个元素是一个类别名称。
            - 如果为 None，则使用 ``metainfo`` 参数、标注文件或类属性 ``METAINFO`` 中的类别信息。

            默认值为 None。
    """ # noqa: E501

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 classes: Union[str, Sequence[str], None] = None):
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)
        metainfo = self._compat_classes(metainfo, classes)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=transforms,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @property
    def img_prefix(self):
        return self.data_prefix['img_path']

    @property
    def CLASSES(self):
        return self._metainfo.get('classes', None)

    @property
    def class_to_idx(self):
        return {cat: i for i, cat in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        gt_labels = np.array(
            [self.get_data_info(i)['gt_label'] for i in range(len(self))])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        return [int(self.get_data_info(idx)['gt_label'])]

    def _compat_classes(self, metainfo, classes):
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmengine.list_from_file(expanduser(classes))
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        elif classes is not None:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if metainfo is None:
            metainfo = {}

        if classes is not None:
            metainfo = {'classes': tuple(class_names), **metainfo}

        return metainfo

    def full_init(self):
        super().full_init()

        if 'categories' in self._metainfo and 'classes' not in self._metainfo:
            categories = sorted(
                self._metainfo['categories'], key=lambda x: x['id'])
            self._metainfo['classes'] = tuple(
                [cat['category_name'] for cat in categories])

    def __repr__(self):
        head = 'Dataset ' + self.__class__.__name__
        body = []
        if self._fully_initialized:
            body.append(f'Number of samples: \t{self.__len__()}')
        else:
            body.append("Haven't been initialized")

        if self.CLASSES is not None:
            body.append(f'Number of categories: \t{len(self.CLASSES)}')

        body.extend(self.extra_repr())

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> List[str]:
        body = []
        body.append(f'Annotation file: \t{self.ann_file}')
        body.append(f'Prefix of images: \t{self.img_prefix}')
        return body
