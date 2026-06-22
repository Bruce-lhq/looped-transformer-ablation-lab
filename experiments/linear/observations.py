"""实验：训练过程基础观测（loss 下降曲线 + 范数误差）。

目的
----
观测 Looped Transformer 在线性回归上的基础训练动力学：
1. loss 下降曲线——验证能正常收敛；
2. y 的范数误差（|pred_norm - true_norm|）随训练的变化——观测预测尺度是否对齐真实尺度。

关键配置（试出来的经验）
------------------------
- loss 观测用了 curriculum learning（``curriculum={'d_x':5, 'seq_len':10, 'duration_ratio':0.5}``）：
  纯 20 维线性回归存在"d_x 之墙"——高维下 loss 极难下降；从低维短序列起步、逐步放大，
  能有效打破这堵墙（见 commit "破除了 linear 数据的叹息之墙"）。观测 loss 是否"正常下降"
  必须先确保训练能收敛，所以这里复用 curriculum 配置。
- norm 观测用 epochs=1000 长训：范数收敛比 loss 慢，需要更长训练才看得到稳态。

运行：``python experiments/linear/observations.py``（图存到 ``figures/linear/``）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

# === 实验 1：loss 下降曲线（带 curriculum，打破 d_x 之墙）===
loss_table = ExperimentTable(
    params_groups=[
        {'experiment_name': 'loss_history',
         'curriculum': {'d_x': 5, 'seq_len': 10, 'duration_ratio': 0.5}},
    ]
)
loss_table.run(result_lists=[(['loss_history'], 'epoch')])
loss_table.plot(figure_size=(8, 6))
(loss_table_dir := ROOT / 'figures' / 'linear').mkdir(parents=True, exist_ok=True)
plt.savefig(loss_table_dir / 'observations_loss.png', dpi=120)
plt.close()

# === 实验 2：y 范数误差（长训观测尺度对齐）===
norm_table = ExperimentTable(
    params_groups=[{'epochs': 1000, 'experiment_name': 'y_norm_error_history', 'print_every': 200}]
)
norm_table.run(result_lists=[(['y_norm_error_history'], 'epoch')])
norm_table.plot(figure_size=(8, 6), compare_experiments=False)
plt.savefig(loss_table_dir / 'observations_norm_error.png', dpi=120)
plt.close()

print('✅ observations 完成，图见 figures/linear/observations_*.png')
