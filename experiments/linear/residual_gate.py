"""实验：残差门控消融（loss 对比 + 门控参数漂移）。

目的
----
Looped Transformer 的残差门控 ``x = a·x_{l-1} + b·x_0`` 中 (a, b) 的类型与初值如何影响收敛，
以及可学习门控在训练中是否会显著漂移。

关键配置（试出来的经验）
------------------------
- ``gate_lr_ratio=100``：门控参数对损失敏感度低，必须用 100 倍于主体的学习率才学得动。
- ``scheduler_type='cosine'``：配合 gate 学习的退火。
- 枚举：fixed (0,0)（无门控）/ learnable_scalar / learnable_vector，初值 (1,0.5)/(1,0)/random。
- 第二张图画 ``*_mean_relative``（相对初始值的变化）：结论是 a/b 漂移幅度都 < 0.2，
  模型倾向维持接近初始值——门控对初值不敏感。

运行：``python experiments/linear/residual_gate.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

out_dir = ROOT / 'figures' / 'linear'
out_dir.mkdir(parents=True, exist_ok=True)

# === 实验 1：各门控配置的 loss 对比 ===
gate_table = ExperimentTable(params_groups=[
    {'experiment_name': 'No Residual Gate',          'residual_gate': (0, 0),     'residual_gate_type': 'fixed'},
    {'experiment_name': 'Residual Gate (1, 0.5)',    'residual_gate': (1, 0.5),   'residual_gate_type': 'learnable_scalar'},
    {'experiment_name': 'Residual Gate (1, 0.5) V',  'residual_gate': (1, 0.5),   'residual_gate_type': 'learnable_vector'},
    {'experiment_name': 'Residual Gate (1, 0)',      'residual_gate': (1, 0),     'residual_gate_type': 'learnable_scalar'},
    {'experiment_name': 'Residual Gate (1, 0) V',    'residual_gate': (1, 0),     'residual_gate_type': 'learnable_vector'},
    {'experiment_name': 'Random Learnable Scalar',   'residual_gate': 'random',   'residual_gate_type': 'learnable_scalar', 'residual_random': (0.9, 0.1)},
    {'experiment_name': 'Random Learnable Vector',   'residual_gate': 'random',   'residual_gate_type': 'learnable_vector', 'residual_random': (0.9, 0.1)},
], manual={'gate_lr_ratio': 100, 'scheduler_type': 'cosine', 'epochs': 50, 'print_every': 25})
gate_table.run(result_lists=[(['loss_history'], 'epoch')])
gate_table.plot(figure_size=(8, 6))
plt.savefig(out_dir / 'residual_gate_loss.png', dpi=120)
plt.close()

# === 实验 2：门控参数相对初始值的漂移（独立模式，每实验双子图）===
gate_drift_table = ExperimentTable(params_groups=[
    {'experiment_name': 'Residual Gate (1, 0.5)',    'residual_gate': (1, 0.5), 'residual_gate_type': 'learnable_scalar'},
    {'experiment_name': 'Residual Gate (1, 0.5) V',  'residual_gate': (1, 0.5), 'residual_gate_type': 'learnable_vector'},
    {'experiment_name': 'Residual Gate (1, 0)',      'residual_gate': (1, 0),   'residual_gate_type': 'learnable_scalar'},
    {'experiment_name': 'Residual Gate (1, 0) V',    'residual_gate': (1, 0),   'residual_gate_type': 'learnable_vector'},
    {'experiment_name': 'Random Learnable Scalar',   'residual_gate': 'random', 'residual_gate_type': 'learnable_scalar', 'residual_random': (0.9, 0.1)},
    {'experiment_name': 'Random Learnable Vector',   'residual_gate': 'random', 'residual_gate_type': 'learnable_vector', 'residual_random': (0.9, 0.1)},
], manual={'gate_lr_ratio': 100, 'scheduler_type': 'cosine', 'epochs': 100, 'print_every': 25})
gate_drift_table.run(result_lists=[
    (['residual_gate_history_a_mean_relative', 'residual_gate_history_b_mean_relative'], 'epoch')
])
gate_drift_table.plot(figure_size=(20, 30), compare_experiments=False, subplot_shape=(-1, 2), suptitle='')
plt.savefig(out_dir / 'residual_gate_drift.png', dpi=120)
plt.close()

print('✅ residual_gate 完成，图见 figures/linear/residual_gate_*.png')
