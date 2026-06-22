"""looped_transformer 包：Looped Transformer 实验框架。

再导出顶层模块与 ``core`` 子包的公开 API，使外部可扁平地使用：
``from looped_transformer import ExperimentTable, ToyModel, default_setup, ...``。
"""

from .core import (
    APE, LearnedAPE, ALiBi, RoPE, MS_UPE,
    SwiGLU,
    AttentionProbe, SinkMetricsProbe,
    lorenz_derivative, rk4_step, lorenz_kernel, create_lorenz_pool,
    HybridOptimizer, Nora, get_nora_optimizer,
    print_vram_usage,
)
from .attention import MultiHeadAttention
from .transformer_block import TransformerBlock
from .toy_model import ToyModel
from .regression import RegressionHead, PredictionLoss, RegressionSolver
from .data_generators import (
    linear_data_generator,
    nonlinear_data_generator,
    lorenz_data_generator,
    clear_lorenz_cache,
)
from .dataloader import dataloader
from .experiment import LoopedTransformerExperiment
from .experiment_table import ExperimentTable
from .default_setup import default_setup

__all__ = [
    # core / position_encoding
    'APE', 'LearnedAPE', 'ALiBi', 'RoPE', 'MS_UPE',
    # core / swiglu
    'SwiGLU',
    # core / probes
    'AttentionProbe', 'SinkMetricsProbe',
    # core / lorenz
    'lorenz_derivative', 'rk4_step', 'lorenz_kernel', 'create_lorenz_pool',
    # core / optimizers
    'HybridOptimizer', 'Nora', 'get_nora_optimizer',
    # core / utils
    'print_vram_usage',
    # 顶层 / 模型
    'MultiHeadAttention', 'TransformerBlock', 'ToyModel',
    # 顶层 / 回归任务
    'RegressionHead', 'PredictionLoss', 'RegressionSolver',
    # 顶层 / 数据
    'linear_data_generator', 'nonlinear_data_generator', 'lorenz_data_generator',
    'clear_lorenz_cache', 'dataloader',
    # 顶层 / 实验框架
    'LoopedTransformerExperiment', 'ExperimentTable', 'default_setup',
]
