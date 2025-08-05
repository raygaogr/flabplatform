import math
import os.path as osp
from typing import Optional, Sequence

from mmengine.fileio import join_path
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from flabplatform.flabcls.registry import HOOKS
from flabplatform.flabcls.structures import DataSample


@HOOKS.register_module()
class VisualizationHook(Hook):
    """分类可视化钩子。

    用于可视化验证和测试预测结果。

    - 如果指定了 ``out_dir``，则忽略所有存储后端，并将图像保存到 ``out_dir``。
    - 如果 ``show`` 为 True，则在窗口中绘制结果图像，请确保能够访问图形界面。

    参数:
        enable (bool): 是否启用此钩子。默认值为 False。
        interval (int): 可视化样本的间隔。默认值为 5000。
        show (bool): 是否显示绘制的图像。默认值为 False。
        out_dir (str, optional): 测试过程中保存绘制图像的目录。
            如果为 None，则使用可视化器的后端处理。默认值为 None。
    """

    def __init__(self,
                 enable=False,
                 interval: int = 5000,
                 show: bool = False,
                 out_dir: Optional[str] = None,
                 **kwargs):
        self._visualizer: Visualizer = Visualizer.get_current_instance()

        self.enable = enable
        self.interval = interval
        self.show = show
        self.out_dir = out_dir

        self.draw_args = {**kwargs, 'show': show}

    def _draw_samples(self,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[DataSample],
                      step: int = 0) -> None:
        """从数据批次中每隔 ``self.interval`` 个样本可视化一次。

        参数:
            batch_idx (int): 当前批次在验证循环中的索引。
            data_batch (dict): 从数据加载器中获取的数据。
            data_samples (Sequence[DataSample]): 模型的输出数据样本。
            step (int): 全局步数值，用于记录。默认值为 0。
        """
        if self.enable is False:
            return

        batch_size = len(data_samples)
        images = data_batch['inputs']
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            image = images[sample_id - start_idx]
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')

            data_sample = data_samples[sample_id - start_idx]
            if 'img_path' in data_sample:
                # osp.basename works on different platforms even file clients.
                sample_name = osp.basename(data_sample.get('img_path'))
            else:
                sample_name = str(sample_id)

            draw_args = self.draw_args
            if self.out_dir is not None:
                draw_args['out_file'] = join_path(self.out_dir,
                                                  f'{sample_name}_{step}.png')

            self._visualizer.visualize_cls(
                image=image,
                data_sample=data_sample,
                step=step,
                name=sample_name,
                **self.draw_args,
            )

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DataSample]) -> None:
        """在验证过程中每隔 ``self.interval`` 个样本进行可视化。

        参数:
            runner (:obj:`Runner`): 验证过程的运行器。
            batch_idx (int): 当前批次在验证循环中的索引。
            data_batch (dict): 从数据加载器中获取的数据。
            outputs (Sequence[DataSample]): 模型的输出数据样本。
        """
        if isinstance(runner.train_loop, EpochBasedTrainLoop):
            step = runner.epoch
        else:
            step = runner.iter

        self._draw_samples(batch_idx, data_batch, outputs, step=step)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DataSample]) -> None:
        """在测试过程中每隔 ``self.interval`` 个样本进行可视化。

        参数:
            runner (:obj:`Runner`): 测试过程的运行器。
            batch_idx (int): 当前批次在测试循环中的索引。
            data_batch (dict): 从数据加载器中获取的数据。
            outputs (Sequence[DataSample]): 模型的输出数据样本。
        """
        self._draw_samples(batch_idx, data_batch, outputs, step=0)
