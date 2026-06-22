"""core 子包：写定的底层积木。

再导出各模块的公开类/函数，使外部可统一从 ``looped_transformer.core`` 或
（经顶层 ``__init__`` 再导出后）直接从 ``looped_transformer`` 访问。
"""

from .position_encoding import APE, LearnedAPE, ALiBi, RoPE, MS_UPE
from .swiglu import SwiGLU
from .probes import AttentionProbe, SinkMetricsProbe
from .lorenz import lorenz_derivative, rk4_step, lorenz_kernel, create_lorenz_pool
from .optimizers import HybridOptimizer, Nora, get_nora_optimizer
from .print_vram_usage import print_vram_usage

__all__ = [
    # position_encoding
    'APE', 'LearnedAPE', 'ALiBi', 'RoPE', 'MS_UPE',
    # swiglu
    'SwiGLU',
    # probes
    'AttentionProbe', 'SinkMetricsProbe',
    # lorenz
    'lorenz_derivative', 'rk4_step', 'lorenz_kernel', 'create_lorenz_pool',
    # optimizers
    'HybridOptimizer', 'Nora', 'get_nora_optimizer',
    # utils
    'print_vram_usage',
]
