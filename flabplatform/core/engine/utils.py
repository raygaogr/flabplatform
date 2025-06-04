from flabplatform.core.config import Config
import os.path as osp



def merge_args(cfg, args):
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
        from .yolorunner import YOLORunnerWarpper
        return YOLORunnerWarpper.from_cfg(cfg)
    else:
        from .mmrunner import MMRunner
        cfg = merge_args(cfg, args)
        return MMRunner.from_cfg(cfg)