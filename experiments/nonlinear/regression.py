"""实验：非线性回归训练观测（MLP 数据：linear + ReLU + linear）。

目的
----
在非线性回归任务（``y = w2·ReLU(w1·x)``）上观测训练动力学：loss、y 范数比、
残差门控漂移。验证 Looped Transformer 对非线性函数的拟合能力。

关键配置（试出来的经验）
------------------------
- ``data_type='nonlinear'``，``function_callable=lambda x: F.relu(x)``，``d_hidden=8``：
  小隐藏层让非线性任务可学（d_hidden 过大反而难收敛）。
- ``pe_type=['alibi']``：线性实验已验证 ALiBi 最优，这里沿用。
- ``residual_gate=(1, 0.5)`` + ``learnable_scalar``：非线性任务下门控需要可学习。
- ``num_eff=5``：非线性用更少有效层（多了梯度不稳）。
- ``init_std='auto'``（=1/√d_model）：GPT-2 风格方差自适应初始化，对非线性很关键。
- ``steps_per_epoch=200``：多步少 epoch，每个 epoch 的指标更平滑。
- 双 Y 轴：loss 左轴，gate a/b 漂移右轴（``'|'`` 分隔）。

运行：``python experiments/nonlinear/regression.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F
from looped_transformer import ExperimentTable

manual = dict(
    data_type='nonlinear',
    function_callable=lambda x: F.relu(x),
    max_seq_len=500,
    scheduled_training=True,
    pe_type=['alibi'],
    residual_gate=(1, 0.5),
    residual_gate_type='learnable_scalar',
    num_eff=5,
    num_blocks=20,
    batch_size=8,
    num_heads=2,
    d_model=128,
    seq_len=400,
    d_hidden=8,
)

loss_table = ExperimentTable(params_groups=[
    {'experiment_name': 'loss_history',
     'epochs': 50,
     'steps_per_epoch': 200,
     'init_std': 'auto',
     'print_every': 5}
], manual=manual)
loss_table.run(result_lists=[
    (['loss_history'], 'epoch'),
    (['y_norm_ratio_history'], 'epoch'),
    # 双 Y 轴：gate a/b 漂移绑右轴
    (['residual_gate_history_a', 'residual_gate_history_b', '|'], 'epoch'),
])
loss_table.plot(compare_experiments=False, subplot_shape=(-1, 1))

out_dir = ROOT / 'figures' / 'nonlinear'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'regression.png', dpi=120)
plt.close()
print('✅ nonlinear regression 完成，图见 figures/nonlinear/regression.png')
