"""实验：线性回归 Curriculum Learning（With vs Without）+ ID/OOD 泛化评估。

目的
----
验证 curriculum learning（从低维短序列起步、逐步放大）能否"打破 d_x 之墙"，并在
训练后评估 ID 基准、X 尺度偏移、序列长度外推三种泛化场景。完整演示训-存-载-评流程。

关键配置（试出来的经验，完全对齐论文设定）
------------------------------------------
- ``x_init='zero'``：绝对零起点，让模型完全靠循环迭代学习（而非直接拷贝输入）。
- ``init_std='auto'``（=1/√d_model）：GPT-2 风格方差自适应初始化。
- ``residual_gate=(1,1)`` + ``fixed``：标准残差注入，不学习门控（剥离外挂）。
- ``optimizer_type='adam', lr=1e-4``，``layer_weight_decay=1.0, seq_weight_decay=1.0``：
  关闭层/序列惩罚，scheduler=None 恒定学习率——这些都是为了"剥离外挂"，看纯架构能力。
- ``scheduled_training=True``：开启 b 的 kick-start（渐进增加 current_blocks）。
- ``epochs=50, steps_per_epoch=200``。
- Curriculum 配置：``{'d_x':5, 'seq_len':10, 'duration_ratio':0.8}``（前 80% 训练逐步放大）。
- ``save_path='auto'``：训练完自动存 checkpoint；评估阶段 ``load_path='auto'`` 自动加载，
  演示断点续训。

评估（3 个场景）
----------------
- ID Baseline：分布内基准；
- OOD Scale x2：``x_scale=2.0`` 测试抗输入扰动；
- OOD Seq Extrapolation：``seq_len_scale=1.2`` 测试长度外推泛化。

运行：``python experiments/curriculum/linear.py``
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

manual_config = dict(
    data_type='linear', d_x=20, seq_len=80, batch_size=64,
    num_blocks=20, num_eff=15, d_model=256, num_heads=8, pe_type=['learned_ape'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    optimizer_type='adam', lr=1e-4,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None,
    epochs=50, steps_per_epoch=200, print_every=1,
    scheduled_training=True,
)

param_groups = [
    {'experiment_name': 'Without Curriculum (Hard Mode)', 'curriculum': {}},
    {'experiment_name': 'With Curriculum (Perfect Path)',
     'curriculum': {'d_x': 5, 'seq_len': 10, 'duration_ratio': 0.8}},
]

eval_configs = [
    {'eval_name': '1_ID_Baseline',           'ood_kwargs': {}},
    {'eval_name': '2_OOD_Scale_x2',          'ood_kwargs': {'x_scale': 2.0}},
    {'eval_name': '3_OOD_Seq_Extrapolation', 'ood_kwargs': {'seq_len_scale': 1.2}},
]
results_lists_eval = [
    (['1_ID_Baseline_sink_scores', '2_OOD_Scale_x2_sink_scores', '3_OOD_Seq_Extrapolation_sink_scores'], 'block'),
    (['1_ID_Baseline_y_norm_ratio', '2_OOD_Scale_x2_y_norm_ratio', '3_OOD_Seq_Extrapolation_y_norm_ratio'], 'experiment'),
    (['1_ID_Baseline_loss', '2_OOD_Scale_x2_loss', '3_OOD_Seq_Extrapolation_loss'], 'experiment'),
]

# === 阶段 1：训练（存 checkpoint）===
train_table = ExperimentTable(params_groups=param_groups, manual={**manual_config, 'save_path': 'auto'})
train_table.run(result_lists=[
    (['loss_history'], 'epoch'),
    (['y_norm_ratio_history'], 'epoch'),
])
train_table.plot(figure_size=(10, 12), compare_experiments=True, subplot_shape=(-1, 1),
                 suptitle='Curriculum Learning Comparison on Looped Transformer')
plt.savefig(out_dir / 'linear_train.png', dpi=120)
plt.close()

# === 阶段 2：加载 checkpoint + ID/OOD 评估 ===
eval_table = ExperimentTable(
    params_groups=[{**p, 'load_path': 'auto'} for p in param_groups],
    manual=manual_config,
)
eval_table.run(result_lists=results_lists_eval, modes=['evaluate'], eval_configs=eval_configs)
eval_table.plot(compare_experiments=False, subplot_shape=(2, -1), suptitle='')
plt.savefig(out_dir / 'linear_eval.png', dpi=120)
plt.close()

print('✅ curriculum/linear 完成，图见 figures/curriculum/linear_{train,eval}.png')
