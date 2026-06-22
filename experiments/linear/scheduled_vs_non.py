"""实验：Scheduled Training vs Non-Scheduled。

目的
----
对比"渐进增加有效层数 current_blocks"（scheduled）与"一开始就用满 num_blocks"（non-scheduled）
对训练收敛的影响。

关键配置（试出来的经验）
------------------------
- ``scheduled_training=True``：current_blocks 从 num_eff 渐增至 num_blocks
  （``current_blocks = min(num_eff + epoch, num_blocks)``）；
- ``scheduled_training=False``（默认）：全程 current_blocks = num_blocks。
- 结论（见 README/报告）：scheduled 在训练初期 loss 更低、更稳，但长期收敛点相近——
  default_setup 默认 ``scheduled_training=False``，因为"用处并不大"，仅在本对比实验中显式开启。

运行：``python experiments/linear/scheduled_vs_non.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

table = ExperimentTable(params_groups=[
    {'scheduled_training': True,  'experiment_name': 'Scheduled Training'},
    {'scheduled_training': False, 'experiment_name': 'Non-Scheduled Training'},
])
table.run(result_lists=[(['loss_history'], 'epoch')])
table.plot(figure_size=(8, 6), compare_experiments=True)

out_dir = ROOT / 'figures' / 'linear'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'scheduled_vs_non.png', dpi=120)
plt.close()
print('✅ scheduled_vs_non 完成，图见 figures/linear/scheduled_vs_non.png')
