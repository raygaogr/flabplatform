from .build_functions import (build_model_from_cfg, build_optimizer_from_cfg,
                              build_runner_from_cfg, build_scheduler_from_cfg)
from .registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', build_func=build_runner_from_cfg, scope="flabplatform.core")
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry('runner constructor', scope="flabplatform.core")
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', scope="flabplatform.core")
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook', scope="flabplatform.core")

# manage all kinds of strategies like `NativeStrategy` and `DDPStrategy`
STRATEGIES = Registry('strategy', scope="flabplatform.core")

# manage data-related modules
DATASETS = Registry('dataset', scope="flabplatform.core")
DATA_SAMPLERS = Registry('data sampler', scope="flabplatform.core")
TRANSFORMS = Registry('transform', scope="flabplatform.core")

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', build_model_from_cfg, scope="flabplatform.core")
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', scope="flabplatform.core")
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry('weight initializer', scope="flabplatform.core")

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', build_func=build_optimizer_from_cfg, scope="flabplatform.core")
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper', scope="flabplatform.core")
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry('optimizer wrapper constructor', scope="flabplatform.core")
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', build_func=build_scheduler_from_cfg, scope="flabplatform.core")

# manage all kinds of metrics
METRICS = Registry('metric', scope="flabplatform.core")
# manage evaluator
EVALUATOR = Registry('evaluator', scope="flabplatform.core")

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', scope="flabplatform.core")

# manage visualizer
VISUALIZERS = Registry('visualizer', scope="flabplatform.core")
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', scope="flabplatform.core")

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor', scope="flabplatform.core")

# manage inferencer
INFERENCERS = Registry('inferencer', scope="flabplatform.core")

# manage function
FUNCTIONS = Registry('function', scope="flabplatform.core")
