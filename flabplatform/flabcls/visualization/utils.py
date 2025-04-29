from typing import Tuple, Union


def get_adaptive_scale(img_shape: Tuple[int, int],
                       min_scale: float = 0.3,
                       max_scale: float = 3.0) -> float:
    """根据图像形状获取自适应缩放比例。

    目标缩放比例取决于图像的短边长度。如果短边长度等于 224，则输出为 1.0。
    根据短边长度输出线性缩放比例。

    您还可以指定最小缩放比例和最大缩放比例，以限制线性缩放范围。

    参数:
        img_shape (Tuple[int, int]): 画布图像的形状。
        min_scale (float): 最小缩放比例。默认为 0.3。
        max_scale (float): 最大缩放比例。默认为 3.0。

    返回:
        float: 自适应缩放比例。
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

def rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size