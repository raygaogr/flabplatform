from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np

import flabplatform.flabcv as flabcv
from .base import BaseTransform
from .builder import TRANSFORMS
from .utils import cache_randomness

Number = Union[int, float]



@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Flip the image & bbox & keypoints & segmentation map. Added or Updated
    keys: flip, flip_direction, img, gt_bboxes, gt_seg_map, and
    gt_keypoints. There are 3 flip modes:

    - ``prob`` is float, ``direction`` is string: the image will be
      ``direction``ly flipped with probability of ``prob`` .
      E.g., ``prob=0.5``, ``direction='horizontal'``,
      then image will be horizontally flipped with probability of 0.5.

    - ``prob`` is float, ``direction`` is list of string: the image will
      be ``direction[i]``ly flipped with probability of
      ``prob/len(direction)``.
      E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
      then image will be horizontally flipped with probability of 0.25,
      vertically with probability of 0.25.

    - ``prob`` is list of float, ``direction`` is list of string:
      given ``len(prob) == len(direction)``, the image will
      be ``direction[i]``ly flipped with probability of ``prob[i]``.
      E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
      'vertical']``, then image will be horizontally flipped with
      probability of 0.3, vertically with probability of 0.5.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - flip
    - flip_direction
    - swap_seg_labels (optional)

    Args:
        prob (float | list[float], optional): The flipping probability.
            Defaults to None.
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
        swap_seg_labels (list, optional): The label pair need to be swapped
            for ground truth, like 'left arm' and 'right arm' need to be
            swapped after horizontal flipping. For example, ``[(1, 5)]``,
            where 1/5 is the label of the left/right arm. Defaults to None.
    """

    def __init__(self,
                 prob: Optional[Union[float, Iterable[float]]] = None,
                 direction: Union[str, Sequence[Optional[str]]] = 'horizontal',
                 swap_seg_labels: Optional[Sequence] = None) -> None:
        if isinstance(prob, list):
            assert mmengine.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmengine.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    def _flip_bbox(self, bboxes: np.ndarray, img_shape: Tuple[int, int],
                   direction: str) -> np.ndarray:
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', and 'diagonal'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        return flipped

    def _flip_keypoints(
        self,
        keypoints: np.ndarray,
        img_shape: Tuple[int, int],
        direction: str,
    ) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagonally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', and 'diagonal'.

        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
        flipped = keypoints.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == 'diagonal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        flipped = np.concatenate([flipped, meta_info], axis=-1)
        return flipped

    def _flip_seg_map(self, seg_map: dict, direction: str) -> np.ndarray:
        """Flip segmentation map horizontally, vertically or diagonally.

        Args:
            seg_map (numpy.ndarray): segmentation map, shape (H, W).
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped segmentation map.
        """
        seg_map = flabcv.imflip(seg_map, direction=direction)
        if self.swap_seg_labels is not None:
            # to handle datasets with left/right annotations
            # like 'Left-arm' and 'Right-arm' in LIP dataset
            # Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/datasets/tools/cv2_aug_transforms.py # noqa:E501
            # Licensed under MIT license
            temp = seg_map.copy()
            assert isinstance(self.swap_seg_labels, (tuple, list))
            for pair in self.swap_seg_labels:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2, \
                    'swap_seg_labels must be a sequence with pair, but got ' \
                    f'{self.swap_seg_labels}.'
                seg_map[temp == pair[0]] = pair[1]
                seg_map[temp == pair[1]] = pair[0]
        return seg_map

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = flabcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'

        return repr_str

