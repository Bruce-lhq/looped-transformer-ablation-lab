"""实验：Sink Padding 消融。

目的
----
测试在序列前端插入不同数量的"全零 sink token"对收敛的影响。Sink token 借鉴自
注意力 sink 现象——给注意力提供一个"垃圾桶"位置，避免其被迫分配给真实 token。

关键配置（试出来的经验）
------------------------
- 枚举 sink_padding ∈ {None, 1, 2, 5, 10}，``epochs=100``。
- 以"无 sink padding"（index=0）为 baseline 看相对差距。
- 结论：在 20 维线性回归任务里，sink padding 对最终 loss **没有决定性正面影响**，
  各组曲线高度重叠——故 default_setup 默认 ``sink_padding=None``。

运行：``python experiments/linear/sink_padding.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

sink_table = ExperimentTable(params_groups=[
    {'experiment_name': 'Without Sink Padding'},
    {'experiment_name': 'With Sink Padding (1)',  'sink_padding': 1},
    {'experiment_name': 'With Sink Padding (2)',  'sink_padding': 2},
    {'experiment_name': 'With Sink Padding (5)',  'sink_padding': 5},
    {'experiment_name': 'With Sink Padding (10)', 'sink_padding': 10},
], manual={'epochs': 100, 'print_every': 20})
sink_table.run(result_lists=[
    (['loss_history'], 'epoch'),
    (['loss_history'], 'epoch', 0),  # 以无 sink padding 为基线
])
sink_table.plot(figure_size=(12, 6))

out_dir = ROOT / 'figures' / 'linear'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'sink_padding.png', dpi=120)
plt.close()
print('✅ sink_padding 完成，图见 figures/linear/sink_padding.png')
