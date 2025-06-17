
from flabplatform.core.registry import DATA_SAMPLERS as ROOT_DATA_SAMPLERS
from flabplatform.core.registry import DATASETS as ROOT_DATASETS
from flabplatform.core.registry import EVALUATOR as ROOT_EVALUATOR
from flabplatform.core.registry import HOOKS as ROOT_HOOKS
from flabplatform.core.registry import INFERENCERS as ROOT_INFERENCERS
from flabplatform.core.registry import LOG_PROCESSORS as ROOT_LOG_PROCESSORS
from flabplatform.core.registry import LOOPS as ROOT_LOOPS
from flabplatform.core.registry import METRICS as ROOT_METRICS
from flabplatform.core.registry import MODEL_WRAPPERS as ROOT_MODEL_WRAPPERS
from flabplatform.core.registry import MODELS as ROOT_MODELS
from flabplatform.core.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as ROOT_OPTIM_WRAPPER_CONSTRUCTORS
from flabplatform.core.registry import OPTIM_WRAPPERS as ROOT_OPTIM_WRAPPERS
from flabplatform.core.registry import OPTIMIZERS as ROOT_OPTIMIZERS
from flabplatform.core.registry import PARAM_SCHEDULERS as ROOT_PARAM_SCHEDULERS
from flabplatform.core.registry import \
    RUNNER_CONSTRUCTORS as ROOT_RUNNER_CONSTRUCTORS
from flabplatform.core.registry import RUNNERS as ROOT_RUNNERS
from flabplatform.core.registry import TASK_UTILS as ROOT_TASK_UTILS
from flabplatform.core.registry import TRANSFORMS as ROOT_TRANSFORMS
from flabplatform.core.registry import VISBACKENDS as ROOT_VISBACKENDS
from flabplatform.core.registry import VISUALIZERS as ROOT_VISUALIZERS
from flabplatform.core.registry import \
    WEIGHT_INITIALIZERS as ROOT_WEIGHT_INITIALIZERS
from flabplatform.core.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=ROOT_RUNNERS, scope='flabplatform.flabseg', locations=[
                   'flabplatform.flabseg.engine'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=ROOT_RUNNER_CONSTRUCTORS, scope='flabplatform.flabseg', locations=['flabplatform.flabseg.engine'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=ROOT_LOOPS, scope='flabplatform.flabseg',
                 locations=['flabplatform.flabseg.engine'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=ROOT_HOOKS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.engine.hooks'])


# manage data-related modules
DATASETS = Registry(
    'dataset', parent=ROOT_DATASETS, scope='flabplatform.flabseg', locations=['flabplatform.flabseg.datasets'])
# DATA_SAMPLERS = Registry('data sampler', parent=ROOT_DATA_SAMPLERS,
#                          scope='flabplatform.flabseg', locations=['flabplatform.flabseg.datasets.samplers'])
TRANSFORMS = Registry(
    'transform',
    parent=ROOT_TRANSFORMS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.datasets.transforms'])

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=ROOT_MODELS, scope='flabplatform.flabseg', locations=[
                  'flabplatform.flabseg.models'])
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=ROOT_MODEL_WRAPPERS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.models'])
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=ROOT_WEIGHT_INITIALIZERS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.models'])

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=ROOT_OPTIMIZERS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.engine.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=ROOT_OPTIM_WRAPPERS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.engine.optimizers'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=ROOT_OPTIM_WRAPPER_CONSTRUCTORS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.engine.optimizers'])
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=ROOT_PARAM_SCHEDULERS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.engine.schedulers'])

# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=ROOT_METRICS, scope='flabplatform.flabseg', locations=['flabplatform.flabseg.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=ROOT_EVALUATOR, scope='flabplatform.flabseg', locations=['flabplatform.flabseg.evaluation'])

# manage task-specific modules like ohem pixel sampler
TASK_UTILS = Registry(
    'task util', parent=ROOT_TASK_UTILS, scope='flabplatform.flabseg', locations=['flabplatform.flabseg.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=ROOT_VISUALIZERS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=ROOT_VISBACKENDS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=ROOT_LOG_PROCESSORS,
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.engine'])

# manage inferencer
INFERENCERS = Registry('inferencer', parent=ROOT_INFERENCERS,
                       scope='flabplatform.flabseg',)

PREDICTORS = Registry(
    'predictor',
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.prediction']
)

VALIDATORS = Registry(
    'validator',
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.validation'],
)

TRAINERS = Registry(
    'trainer',
    scope='flabplatform.flabseg',
    locations=['flabplatform.flabseg.train'],
)

