import inspect
import math
import re
from enum import EnumMeta
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import flabplatform.flabcv as flabcv
import numpy as np
import torchvision
from flabplatform.flabcv.transforms import BaseTransform
from flabplatform.flabcv.transforms.utils import cache_randomness
from torchvision.transforms.transforms import InterpolationMode

from flabplatform.flabcls.registry import TRANSFORMS

try:
    import albumentations
except ImportError:
    albumentations = None


def _str_to_torch_dtype(t: str):
    """mapping str format dtype to torch.dtype."""
    import torch  # noqa: F401,F403
    return eval(f'torch.{t}')


def _interpolation_modes_from_str(t: str):
    """mapping str format to Interpolation."""
    t = t.lower()
    inverse_modes_mapping = {
        'nearest': InterpolationMode.NEAREST,
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'box': InterpolationMode.BOX,
        'hammimg': InterpolationMode.HAMMING,
        'lanczos': InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[t]


class TorchVisonTransformWrapper:

    def __init__(self, transform, *args, **kwargs):
        if 'interpolation' in kwargs and isinstance(kwargs['interpolation'],
                                                    str):
            kwargs['interpolation'] = _interpolation_modes_from_str(
                kwargs['interpolation'])
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], str):
            kwargs['dtype'] = _str_to_torch_dtype(kwargs['dtype'])
        self.t = transform(*args, **kwargs)

    def __call__(self, results):
        results['img'] = self.t(results['img'])
        return results

    def __repr__(self) -> str:
        return f'TorchVision{repr(self.t)}'


def register_vision_transforms() -> List[str]:
    """Register transforms in ``torchvision.transforms`` to the ``TRANSFORMS``
    registry.

    Returns:
        List[str]: A list of registered transforms' name.
    """
    vision_transforms = []
    for module_name in dir(torchvision.transforms):
        if not re.match('[A-Z]', module_name):
            # must startswith a capital letter
            continue
        _transform = getattr(torchvision.transforms, module_name)
        if inspect.isclass(_transform) and callable(
                _transform) and not isinstance(_transform, (EnumMeta)):
            from functools import partial
            TRANSFORMS.register_module(
                module=partial(
                    TorchVisonTransformWrapper, transform=_transform),
                name=f'torchvision/{module_name}')
            vision_transforms.append(f'torchvision/{module_name}')
    return vision_transforms


# register all the transforms in torchvision by using a transform wrapper
VISION_TRANSFORMS = register_vision_transforms()


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Crop the given Image at a random location.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        crop_size (int | Sequence): Desired output size of the crop. If
            crop_size is an int instead of sequence like (h, w), a square crop
            (crop_size, crop_size) is made.
        padding (int | Sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (bool): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:

            - ``constant``: Pads with a constant value, this value is specified
              with pad_val.
            - ``edge``: pads with the last value at the edge of the image.
            - ``reflect``: Pads with reflection of image without repeating the
              last value on the edge. For example, padding [1, 2, 3, 4]
              with 2 elements on both sides in reflect mode will result
              in [3, 2, 1, 2, 3, 4, 3, 2].
            - ``symmetric``: Pads with reflection of image repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with
              2 elements on both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 crop_size: Union[Sequence, int],
                 padding: Optional[Union[Sequence, int]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[Number, Sequence[Number]] = 0,
                 padding_mode: str = 'constant'):
        if isinstance(crop_size, Sequence):
            assert len(crop_size) == 2
            assert crop_size[0] > 0 and crop_size[1] > 0
            self.crop_size = crop_size
        else:
            assert crop_size > 0
            self.crop_size = (crop_size, crop_size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        target_h, target_w = self.crop_size
        if w == target_w and h == target_h:
            return 0, 0, h, w
        elif w < target_w or h < target_h:
            target_w = min(w, target_w)
            target_h = min(h, target_h)

        offset_h = np.random.randint(0, h - target_h + 1)
        offset_w = np.random.randint(0, w - target_w + 1)

        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        if self.padding is not None:
            img = flabcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - img.shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - img.shape[1]) / 2)

            img = flabcv.impad(
                img,
                padding=(w_pad, h_pad, w_pad, h_pad),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = flabcv.imcrop(
            img,
            np.array([
                offset_w,
                offset_h,
                offset_w + target_w - 1,
                offset_h + target_h - 1,
            ]))
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', padding={self.padding}'
        repr_str += f', pad_if_needed={self.pad_if_needed}'
        repr_str += f', pad_val={self.pad_val}'
        repr_str += f', padding_mode={self.padding_mode})'
        return repr_str
