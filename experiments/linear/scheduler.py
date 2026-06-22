"""实验：学习率调度器对比（None / Cosine / Step）。

目的
----
对比无调度器、Cosine 退火、StepLR 三者对 loss / y 范数的影响。

关键配置（试出来的经验）
------------------------
这是一个"配置覆写"的典型样例——``manual`` 设全局默认（``scheduler_type=None``、
``lr_scale=0.01``），再在 ``params_groups`` 里逐实验覆写：
- Cosine 实验覆写 ``lr_scale=0``（即 eta_min=0，完全退火到 0）；
- Step 实验覆写 ``lr_scale=0.1, step_size_scheduler=90``（每 90 步衰减到 0.1）。
- 其余共用：``d_model=32, num_heads=1``（小模型加速）、``pe_type=['alibi']``（已验证最优 PE）、
  ``seq_len=400``（长序列）、``optimizer_type='adamw', wd_adamw=0.2``、可学习标量门控、
  ``layer_weight_decay=0.8, seq_weight_decay=0.9``。
- 以"No Scheduler"（index=0）为 baseline 看 loss / y_pred_norm 的相对差。

运行：``python experiments/linear/scheduler.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

experiment_table = ExperimentTable(
    params_groups=[
        {'experiment_name': 'No Scheduler'},
        {'scheduler_type': 'cosine', 'experiment_name': 'Cosine Scheduler', 'lr_scale': 0},
        {'scheduler_type': 'step',   'experiment_name': 'Step Scheduler',   'lr_scale': 0.1, 'step_size_scheduler': 90},
    ],
    manual={
        'epochs': 200, 'lr': 1e-3, 'optimizer_type': 'adamw', 'pe_type': ['alibi'],
        'wd_adamw': 0.2, 'd_model': 32, 'num_heads': 1, 'max_seq_len': 500, 'seq_len': 400, 'batch_size': 64, 'd_x': 20,
        'num_blocks': 20, 'num_eff': 15,
        'residual_gate': (1, 1), 'residual_gate_type': 'learnable_scalar', 'gate_lr_ratio': 100,
        'scheduler_type': None, 'lr_scale': 0.01, 'step_size_scheduler': 10,
        'print_every': 50,
        'layer_weight_decay': 0.8, 'seq_weight_decay': 0.9,
    }
)
experiment_table.run(result_lists=[
    (['loss_history'], 'epoch', 0),
    (['y_pred_norm_history'], 'epoch', 0),
    (['y_true_norm_history'], 'epoch'),
])
experiment_table.plot(figure_size=(18, 6))

out_dir = ROOT / 'figures' / 'linear'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'scheduler.png', dpi=120)
plt.close()
print('✅ scheduler 完成，图见 figures/linear/scheduler.png')
