"""实验：5 种位置编码对比 + MS-UPE 与 Learned APE 叠加。

目的
----
在 Looped Transformer（权重共享、循环迭代）设定下，对比位置编码注入位置的影响：
- 输入端加法（APE / LearnedAPE）
- Q/K 旋转（RoPE）
- Q/K 加法多尺度（MS-UPE，自创）
- score 矩阵线性偏置（ALiBi）
以及 MS-UPE 与 LearnedAPE 叠加是否优于单独使用。

关键配置（试出来的经验）
------------------------
- ``epochs=100``：PE 差异需要足够长训练才显现。
- 以 **ALiBi（index=2）为 baseline** 画相对 loss：实测 ALiBi 综合最优，作为基准看其他 PE
  的相对差距最直观。
- 结论：在循环架构里，score/QK 层面注入（ALiBi/RoPE）优于输入端（LearnedAPE）；
  MS-UPE+LearnedAPE 叠加反而不如单独 LearnedAPE（PE 叠加无益）。

运行：``python experiments/linear/pe_compare.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

pe_table = ExperimentTable(params_groups=[
    {'pe_type': ['learned_ape'],            'experiment_name': 'Learned APE'},
    {'pe_type': ['ape'],                    'experiment_name': 'APE'},
    {'pe_type': ['alibi'],                  'experiment_name': 'ALiBi'},
    {'pe_type': ['rope'],                   'experiment_name': 'RoPE'},
    {'pe_type': ['ms_upe'],                 'experiment_name': 'MS-UPE'},
    {'pe_type': ['ms_upe', 'learned_ape'],  'experiment_name': 'MS-UPE + Learned APE'},
], manual={'epochs': 100, 'print_every': 20})
pe_table.run(result_lists=[
    (['loss_history'], 'epoch'),
    (['loss_history'], 'epoch', 2),  # 以 ALiBi（index=2）为基线
])
pe_table.plot(figure_size=(12, 6))

out_dir = ROOT / 'figures' / 'linear'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'pe_compare.png', dpi=120)
plt.close()
print('✅ pe_compare 完成，图见 figures/linear/pe_compare.png')
