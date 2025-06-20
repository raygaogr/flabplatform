from flabplatform.core.registry import DATA_SAMPLERS as ROOT_DATA_SAMPLERS
from flabplatform.core.registry import DATASETS as ROOT_DATASETS
from flabplatform.core.registry import EVALUATOR as ROOT_EVALUATOR

from flabplatform.core.registry import HOOKS as ROOT_HOOKS
from flabplatform.core.registry import LOG_PROCESSORS as ROOT_LOG_PROCESSORS
from flabplatform.core.registry import LOOPS as ROOT_LOOPS
from flabplatform.core.registry import METRICS as ROOT_METRICS

from flabplatform.core.registry import MODEL_WRAPPERS as ROOT_MODEL_WRAPPERS
from flabplatform.core.registry import MODELS as ROOT_MODELS

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
    'RUNNERS', 'RUNNER_CONSTRUCTORS',  'HOOKS',
    'DATASETS', 'DATA_SAMPLERS', 'MODELS',
    'MODEL_WRAPPERS', 'WEIGHT_INITIALIZERS', 'BATCH_AUGMENTS', 'TASK_UTILS',
    'METRICS', 'EVALUATORS', 'VISUALIZERS', 'VISBACKENDS'
]
#######################################################################
#                   检测任务中engine相关的注册器                      #
#######################################################################

# 检测任务中的执行器，用于整个流程的封装和执行
RUNNERS = Registry(
    'runner',
    parent=ROOT_RUNNERS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.engine'],
)
# 检测任务中的执行器构造器，用于定义如何初始化执行器
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=ROOT_RUNNER_CONSTRUCTORS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.engine'],
)
# # 循环，用于定义训练或测试过程
# LOOPS = Registry(
#     'loop',
#     parent=ROOT_LOOPS,
#     scope='flabplatform.flabdet',
#     locations=['flabplatform.flabdet.engine'],
# )

TRAINERS = Registry(
    'trainer',
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.train'],
)

VALIDATORS = Registry(
    'validator',
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.validation'],
)


# 钩子，用于在运行过程中添加额外的功能，如`CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=ROOT_HOOKS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.utils.mmdet.hooks'],
)
# # 日志处理器，用于处理标量日志数据
# LOG_PROCESSORS = Registry(
#     'log processor',
#     scope='flabplatform.flabdet',
#     parent=ROOT_LOG_PROCESSORS,
#     locations=['flabplatform.flabdet.engine'],
# )
# # 优化器，用于优化模型权重，如`SGD`和`Adam`
# OPTIMIZERS = Registry(
#     'optimizer',
#     scope='flabplatform.flabdet',
#     parent=ROOT_OPTIMIZERS,
#     locations=['flabplatform.flabdet.engine'],
# )
# # 优化器包装器，用于增强优化过程
# OPTIM_WRAPPERS = Registry(
#     'optimizer_wrapper',
#     scope='flabplatform.flabdet',
#     parent=ROOT_OPTIM_WRAPPERS,
#     locations=['flabplatform.flabdet.engine'],
# )

# # 参数调度器，用于动态调整优化参数
# PARAM_SCHEDULERS = Registry(
#     'parameter scheduler',
#     scope='flabplatform.flabdet',
#     parent=ROOT_PARAM_SCHEDULERS,
#     locations=['flabplatform.flabdet.engine'],
# )

#######################################################################
#                   检测任务中datasets相关的注册器                      #
#######################################################################

# 检测任务中的数据集，如`ImageNet`和`CIFAR10`
DATASETS = Registry(
    'dataset',
    parent=ROOT_DATASETS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.datasets'],
)
# 数据集采样器，用于采样数据集
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=ROOT_DATA_SAMPLERS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.datasets.mmdet.samplers'],
)

# 数据集的转换，用于处理数据集中的样本
TRANSFORMS = Registry(
    'transform',
    parent=ROOT_TRANSFORMS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.datasets.mmdet.transforms'],
)

#######################################################################
#                   检测任务中models相关的注册器                        #
#######################################################################

# 模型，用于定义神经网络结构
MODELS = Registry(
    'model',
    parent=ROOT_MODELS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.models'],
)
# 模型包装器，用于增强模型的功能，如`MMDistributedDataParallel`
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=ROOT_MODEL_WRAPPERS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.models'],
)
# 权重初始化方法，如`uniform`和`xavier`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=ROOT_WEIGHT_INITIALIZERS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.models'],
)
# 批量增强，如`Mixup`和`CutMix`
BATCH_AUGMENTS = Registry(
    'batch augment',
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.models'],
)
# 任务工具，用于定义任务相关的模块，例如锚框生成器和框编码器
TASK_UTILS = Registry(
    'task util',
    parent=ROOT_TASK_UTILS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.models'],
)

#######################################################################
#               检测任务中evaluation相关的注册器                        #
#######################################################################

# 评估指标，用于评估模型预测结果
METRICS = Registry(
    'metric',
    parent=ROOT_METRICS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.validation.mmdet'],
)
# 评估器，用于定义评估过程
EVALUATORS = Registry(
    'evaluator',
    parent=ROOT_EVALUATOR,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.validation.mmdet'],
)

#######################################################################
#                      检测任务中visualization相关注册器                      #
#######################################################################
PREDICTORS = Registry(
    'predictor',
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.prediction']
)


# 可视化器，用于展示任务相关的结果
VISUALIZERS = Registry(
    'visualizer',
    parent=ROOT_VISUALIZERS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.utils.mmdet.visualization'],
)

# 可视化后端，用于保存可视化结果，如`Tensorboard`和`Visdom`
VISBACKENDS = Registry(
    'vis_backend',
    parent=ROOT_VISBACKENDS,
    scope='flabplatform.flabdet',
    locations=['flabplatform.flabdet.utils.mmdet.visualization'],
)
