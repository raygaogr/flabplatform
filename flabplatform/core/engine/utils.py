from flabplatform.core.config import Config, ConfigDict
from copy import deepcopy
import os.path as osp
from flabplatform.flabdet.utils.yolos import SETTINGS
import os
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.device import is_cuda_available, is_musa_available
from mmengine.dist import get_rank, sync_random_seed
from flabplatform.core.logging import print_log
from mmengine.utils import digit_version, is_list_of
from mmengine.utils.dl_utils import TORCH_VERSION


def calc_dynamic_intervals(
    start_interval: int,
    dynamic_interval_list: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[int], List[int]]:
    """Calculate dynamic intervals.

    Args:
        start_interval (int): The interval used in the beginning.
        dynamic_interval_list (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.

    Returns:
        Tuple[List[int], List[int]]: a list of milestone and its corresponding
        intervals.
    """
    if dynamic_interval_list is None:
        return [0], [start_interval]

    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


def set_random_seed(seed: Optional[int] = None,
                    deterministic: bool = False,
                    diff_rank_seed: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
        diff_rank_seed (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Defaults to False.
    """
    if seed is None:
        seed = sync_random_seed()

    if diff_rank_seed:
        rank = get_rank()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    elif is_musa_available():
        torch.musa.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print_log(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm',
                logger='current',
                level=logging.WARNING)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
            torch.use_deterministic_algorithms(True)
    return seed


def _get_batch_size(dataloader: dict):
    if isinstance(dataloader, dict):
        if 'batch_size' in dataloader:
            return dataloader['batch_size']
        elif ('batch_sampler' in dataloader
              and 'batch_size' in dataloader['batch_sampler']):
            return dataloader['batch_sampler']['batch_size']
        else:
            raise ValueError('Please set batch_size in `Dataloader` or '
                             '`batch_sampler`')
    elif isinstance(dataloader, DataLoader):
        return dataloader.batch_sampler.batch_size
    else:
        raise ValueError('dataloader should be a dict or a Dataloader '
                         f'instance, but got {type(dataloader)}')



def merge_args(cfg, args):
    cfg.launcher = None

    # if args.work_dir is not None:
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    cfg.work_dir = osp.join('./res', osp.splitext(osp.basename(args.config))[0])

    # # enable automatic-mixed-precision training
    # if args.amp is True:
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.loss_scale = 'dynamic'

    # # enable automatically scaling LR
    # if args.auto_scale_lr:
    #     if 'auto_scale_lr' in cfg and \
    #             'enable' in cfg.auto_scale_lr and \
    #             'base_batch_size' in cfg.auto_scale_lr:
    #         cfg.auto_scale_lr.enable = True
    #     else:
    #         raise RuntimeError('Can not find "auto_scale_lr" or '
    #                            '"auto_scale_lr.enable" or '
    #                            '"auto_scale_lr.base_batch_size" in your'
    #                            ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    # if args.resume == 'auto':
    # cfg.resume = True
    # cfg.load_from = None
    # elif args.resume is not None:
    #     cfg.resume = True
    #     cfg.load_from = args.resume

    return cfg


def create_runner(args):
    """Create a runner instance."""
    cfg = Config.fromfile(args.config)
    modelname = cfg.training['algoParams']['model']['type']

    if 'yolo' in modelname:
        datasets_dir = os.path.join(cfg.commonParams["datasets"]["rootDir"])
        save_dir = cfg.commonParams["outputDir"]
        SETTINGS.update(dict(datasets_dir=datasets_dir))
        SETTINGS.update(dict(runs_dir=save_dir))
        from .yolorunner import YOLORunnerWarpper
        return YOLORunnerWarpper.from_cfg(cfg)
    else:
        from .mmrunner import MMRunner
        cfg = merge_args(cfg, args)
        return MMRunner.from_cfg(cfg)