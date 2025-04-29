from typing import Optional, Sequence

import cv2
import numpy as np
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from flabplatform.flabcls.registry import VISUALIZERS
from flabplatform.flabcls.structures import DataSample
from .utils import get_adaptive_scale, rescale_size


@VISUALIZERS.register_module()
class UniversalVisualizer(Visualizer):
    """通用可视化器，用于多任务可视化。

    参数:
        name (str): 实例的名称。默认为 'visualizer'。
        image (np.ndarray, optional): 要绘制的原始图像。格式应为 RGB。默认为 None。
        vis_backends (list, optional): 可视化后端配置列表。默认为 None。
        save_dir (str, optional): 所有存储后端的保存文件目录。如果为 None，则后端存储不会保存任何数据。
        fig_save_cfg (dict): 保存图形的关键字参数。默认为空字典。
        fig_show_cfg (dict): 显示图形的关键字参数。默认为空字典。
    """
    DEFAULT_TEXT_CFG = {
        'family': 'monospace',
        'color': 'white',
        'bbox': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
        'verticalalignment': 'top',
        'horizontalalignment': 'left',
    }

    @master_only
    def visualize_cls(self,
                      image: np.ndarray,
                      data_sample: DataSample,
                      classes: Optional[Sequence[str]] = None,
                      draw_gt: bool = True,
                      draw_pred: bool = True,
                      draw_score: bool = True,
                      resize: Optional[int] = None,
                      rescale_factor: Optional[float] = None,
                      text_cfg: dict = dict(),
                      show: bool = False,
                      wait_time: float = 0,
                      out_file: Optional[str] = None,
                      name: str = '',
                      step: int = 0) -> None:
        """可视化图像分类结果。

        此方法将在输入图像上绘制文本框，以可视化图像分类的信息，例如真实标签和预测标签。

        参数:
            image (np.ndarray): 要绘制的图像。格式应为 RGB。
            data_sample (:obj:`DataSample`): 图像的标注信息。
            classes (Sequence[str], optional): 类别名称。默认为 None。
            draw_gt (bool): 是否绘制真实标签。默认为 True。
            draw_pred (bool): 是否绘制预测标签。默认为 True。
            draw_score (bool): 是否绘制预测类别的分数。默认为 True。
            resize (int, optional): 在可视化之前将图像的短边调整为指定长度。默认为 None。
            rescale_factor (float, optional): 在可视化之前按比例缩放图像。默认为 None。
            text_cfg (dict): 额外的文本设置，接受 :meth:`mmengine.Visualizer.draw_texts` 的参数。默认为空字典。
            show (bool): 是否在窗口中显示绘制的图像，请确认您可以访问图形界面。默认为 False。
            wait_time (float): 显示时间（秒）。默认为 0，表示“永久”。
            out_file (str, optional): 保存可视化结果的额外路径。如果指定，visualizer 只会将结果图像保存到 out_file，而忽略其存储后端。默认为 None。
            name (str): 图像标识符。在使用 visualizer 的存储后端保存或显示图像时很有用。默认为空字符串。
            step (int): 全局步数值。在使用存储后端记录同一图像的一系列可视化结果时很有用。默认为 0。

        返回:
            np.ndarray: 可视化后的图像。
        """
        if self.dataset_meta is not None:
            classes = classes or self.dataset_meta.get('classes', None)

        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = cv2.resize(image, (resize, resize * h // w))
            else:
                image = cv2.resize(image, (resize * w // h, resize))
        elif rescale_factor is not None:
            h, w = image.shape[:2]
            new_size, _ = rescale_size((w, h), rescale_factor, return_scale=True)
            image = cv2.resize(image, new_size)

        texts = []
        self.set_image(image)

        if draw_gt and 'gt_label' in data_sample:
            idx = data_sample.gt_label.tolist()
            class_labels = [''] * len(idx)
            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]
            labels = [str(idx[i]) + class_labels[i] for i in range(len(idx))]
            prefix = 'Ground truth: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        if draw_pred and 'pred_label' in data_sample:
            idx = data_sample.pred_label.tolist()
            score_labels = [''] * len(idx)
            class_labels = [''] * len(idx)
            if draw_score and 'pred_score' in data_sample:
                score_labels = [
                    f', {data_sample.pred_score[i].item():.2f}' for i in idx
                ]

            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]

            labels = [
                str(idx[i]) + score_labels[i] + class_labels[i]
                for i in range(len(idx))
            ]
            prefix = 'Prediction: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            'size': int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            '\n'.join(texts),
            **text_cfg,
        )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            cv2.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img