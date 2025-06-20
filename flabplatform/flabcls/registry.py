from flabplatform.core.registry import DATA_SAMPLERS as ROOT_DATA_SAMPLERS
from flabplatform.core.registry import DATASETS as ROOT_DATASETS
from flabplatform.core.registry import EVALUATOR as ROOT_EVALUATOR
from flabplatform.core.registry import HOOKS as ROOT_HOOKS
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

__all__ = [
    'RUNNERS', 'RUNNER_CONSTRUCTORS', 'LOOPS', 'HOOKS', 'LOG_PROCESSORS',
    'OPTIMIZERS', 'OPTIM_WRAPPERS', 'OPTIM_WRAPPER_CONSTRUCTORS',
    'PARAM_SCHEDULERS', 'DATASETS', 'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS',
    'MODEL_WRAPPERS', 'WEIGHT_INITIALIZERS', 'BATCH_AUGMENTS', 'TASK_UTILS',
    'METRICS', 'EVALUATORS', 'VISUALIZERS', 'VISBACKENDS'
]

#######################################################################
#                  分类任务中engine相关的注册器                         #
#######################################################################

# 分类任务中的执行器，用于整个流程的封装和执行
RUNNERS = Registry(
    'runner',
    parent=ROOT_RUNNERS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.engine'],
)
# 分类任务中的执行器构造器，用于定义如何初始化执行器
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=ROOT_RUNNER_CONSTRUCTORS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.engine'],
)
# 循环，用于定义训练或测试过程
LOOPS = Registry(
    'loop',
    parent=ROOT_LOOPS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.engine'],
)
# 钩子，用于在运行过程中添加额外的功能，如`CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=ROOT_HOOKS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.engine.hooks'],
)
# 日志处理器，用于处理标量日志数据
LOG_PROCESSORS = Registry(
    'log processor',
    scope='cloudplatform.flabcls',
    parent=ROOT_LOG_PROCESSORS,
    locations=['cloudplatform.flabcls.engine'],
)
# 优化器，用于优化模型权重，如`SGD`和`Adam`
OPTIMIZERS = Registry(
    'optimizer',
    scope='cloudplatform.flabcls',
    parent=ROOT_OPTIMIZERS,
    locations=['cloudplatform.flabcls.engine'],
)
# 优化器包装器，用于增强优化过程
OPTIM_WRAPPERS = Registry(
    'optimizer_wrapper',
    scope='cloudplatform.flabcls',
    parent=ROOT_OPTIM_WRAPPERS,
    locations=['cloudplatform.flabcls.engine'],
)
# 优化器包装器构造器，用于自定义优化器的超参数
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    scope='cloudplatform.flabcls',
    parent=ROOT_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['cloudplatform.flabcls.engine'],
)
# 参数调度器，用于动态调整优化参数
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    scope='cloudplatform.flabcls',
    parent=ROOT_PARAM_SCHEDULERS,
    locations=['cloudplatform.flabcls.engine'],
)

#######################################################################
#                   分类任务中datasets相关的注册器                      #
#######################################################################

# 分类任务中的数据集，如`ImageNet`和`CIFAR10`
DATASETS = Registry(
    'dataset',
    parent=ROOT_DATASETS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.datasets'],
)
# 数据集采样器，用于采样数据集
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=ROOT_DATA_SAMPLERS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.datasets.samplers'],
)
# 数据集的转换，用于处理数据集中的样本
TRANSFORMS = Registry(
    'transform',
    parent=ROOT_TRANSFORMS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.datasets.transforms'],
)

#######################################################################
#                   分类任务中models相关的注册器                        #
#######################################################################

# 模型，用于定义神经网络结构
MODELS = Registry(
    'model',
    parent=ROOT_MODELS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.models'],
)
# 模型包装器，用于增强模型的功能，如`MMDistributedDataParallel`
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=ROOT_MODEL_WRAPPERS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.models'],
)
# 权重初始化方法，如`uniform`和`xavier`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=ROOT_WEIGHT_INITIALIZERS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.models'],
)
# 批量增强，如`Mixup`和`CutMix`
BATCH_AUGMENTS = Registry(
    'batch augment',
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.models'],
)
# 任务工具，用于定义任务相关的模块，例如锚框生成器和框编码器
TASK_UTILS = Registry(
    'task util',
    parent=ROOT_TASK_UTILS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.models'],
)

#######################################################################
#               分类任务中evaluation相关的注册器                        #
#######################################################################

# 评估指标，用于评估模型预测结果
METRICS = Registry(
    'metric',
    parent=ROOT_METRICS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.evaluation'],
)
# 评估器，用于定义评估过程
EVALUATORS = Registry(
    'evaluator',
    parent=ROOT_EVALUATOR,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.evaluation'],
)

#######################################################################
#                      分类任务中visualization相关注册器                      #
#######################################################################

# 可视化器，用于展示任务相关的结果
VISUALIZERS = Registry(
    'visualizer',
    parent=ROOT_VISUALIZERS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.visualization'],
)
# 可视化后端，用于保存可视化结果，如`Tensorboard`和`Visdom`
VISBACKENDS = Registry(
    'vis_backend',
    parent=ROOT_VISBACKENDS,
    scope='cloudplatform.flabcls',
    locations=['cloudplatform.flabcls.visualization'],
)
