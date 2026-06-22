"""实验：Lorenz 吸引子 Curriculum Learning + ID/OOD 泛化评估。

目的
----
在混沌动力学系统（Lorenz 方程，下一帧状态预测）上验证 curriculum learning，并评估
ID 基准、物理参数偏移（rho 漂移）、序列长度外推三种泛化场景。

关键配置（试出来的经验）
------------------------
- ``data_type='lorenz'``，``d_x=3, d_y=3``：Lorenz 状态空间 (X,Y,Z) 是固定物理维度，
  **不能像 linear/nonlinear 那样对 d_x 做课程**——所以 curriculum 只演进 seq_len。
- ``lorenz_kwargs=dict(dt=0.01, burn_in=500)``：burn-in 500 步让系统进入吸引子区域再采样。
- ``seq_len=300``（150 对 (x,y)），``max_seq_len=500``。
- curriculum：``{'seq_len': 20, 'duration_ratio': 0.8}``——从极短的 20 步起步，前 80%
  训练逐步拉长到 300 步（降低初始长序列的积分关联难度）。
- 数据池：``load_lorenz_from`` 指向离线池。**池不存在时自动 fallback 到实时 RK4 生成**
  （带 warning）。如需快速迭代，可先跑 ``create_lorenz_pool(pool_size=200000, traj_len=1000, ...)``
  生成高密度池（约 2.4GB，gitignored）。
- 模型/优化器与 linear/nonlinear 对齐论文：``x_init='zero', init_std='auto'，
  residual_gate=(1,1) fixed, adam lr=1e-4``，关闭 weight_decay，scheduled_training kick-start。

评估（3 个场景，针对混沌特征量身定制）
--------------------------------------
- ID Baseline：分布内单步预测；
- OOD Param Shift：``rho_shift=5.0``（rho: 28→33），测试是否学到了隐式微分算子而非背几何；
- OOD Seq Extrapolation：``seq_len_scale=1.2``，更长历史窗口的泛化边界。

运行：``python experiments/curriculum/lorenz.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

out_dir = ROOT / 'figures' / 'curriculum'
out_dir.mkdir(parents=True, exist_ok=True)

# 离线数据池路径（不存在时 lorenz_data_generator 会 fallback 到实时生成）
MY_LORENZ_POOL_PATH = 'data/lorenz/length_1000_dt0.01_sigma10.0_beta2.7_rho28.0.pth'

manual_config = dict(
    data_type='lorenz',
    d_x=3, d_y=3,                # Lorenz 状态空间 (X,Y,Z) 的固定物理维度
    max_seq_len=500, seq_len=300, batch_size=64,
    num_blocks=20, num_eff=15, d_model=256, num_heads=8, pe_type=['learned_ape'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    lorenz_kwargs=dict(dt=0.01, burn_in=500),
    load_lorenz_from=MY_LORENZ_POOL_PATH,
    optimizer_type='adam', lr=1e-4,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None,
    epochs=50, steps_per_epoch=50, print_every=2,
    scheduled_training=True,
)

param_groups = [
    {'experiment_name': 'Lorenz Without Curriculum (Hard Mode)', 'curriculum': {}},
    # 注意：lorenz 的 d_x=3 是物理维度不能变，curriculum 只演进 seq_len
    {'experiment_name': 'Lorenz With Curriculum (Perfect Path)',
     'curriculum': {'seq_len': 20, 'duration_ratio': 0.8}},
]

eval_configs = [
    {'eval_name': '1_ID_Baseline',           'ood_kwargs': {}},
    {'eval_name': '2_OOD_Param_Shift',       'ood_kwargs': {'rho_shift': 5.0}},
    {'eval_name': '3_OOD_Seq_Extrapolation', 'ood_kwargs': {'seq_len_scale': 1.2}},
]
results_lists_eval = [
    (['1_ID_Baseline_sink_scores', '2_OOD_Param_Shift_sink_scores', '3_OOD_Seq_Extrapolation_sink_scores'], 'block'),
    (['1_ID_Baseline_y_norm_ratio', '2_OOD_Param_Shift_y_norm_ratio', '3_OOD_Seq_Extrapolation_y_norm_ratio'], 'experiment'),
    (['1_ID_Baseline_loss', '2_OOD_Param_Shift_loss', '3_OOD_Seq_Extrapolation_loss'], 'experiment'),
]

# === 阶段 1：训练（存 checkpoint）===
train_table = ExperimentTable(params_groups=param_groups, manual={**manual_config, 'save_path': 'auto'})
train_table.run(result_lists=[
    (['loss_history'], 'epoch'),
    (['y_norm_ratio_history'], 'epoch'),
])
train_table.plot(subplot_shape=(-1, 1),
                 suptitle='Lorenz Attractor Dynamic Curriculum Learning Comparison')
plt.savefig(out_dir / 'lorenz_train.png', dpi=120)
plt.close()

# === 阶段 2：加载 checkpoint + ID/OOD 评估 ===
eval_table = ExperimentTable(
    params_groups=[{**p, 'load_path': 'auto'} for p in param_groups],
    manual=manual_config,
)
eval_table.run(result_lists=results_lists_eval, modes=['evaluate'], eval_configs=eval_configs, parallel_workers=2)
eval_table.plot(compare_experiments=False, subplot_shape=(2, -1),
                suptitle='OOD Evaluation Metrics on Chaotic Systems')
plt.savefig(out_dir / 'lorenz_eval.png', dpi=120)
plt.close()

print('✅ curriculum/lorenz 完成，图见 figures/curriculum/lorenz_{train,eval}.png')
