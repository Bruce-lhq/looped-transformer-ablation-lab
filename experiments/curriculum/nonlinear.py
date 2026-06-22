"""实验：非线性回归 Curriculum Learning + ID/OOD 泛化评估。

目的
----
在非线性任务（``y = √2·ReLU(w2·ReLU(w1·x))``，系数 √2 使方差期望为 1）上验证
curriculum learning 的作用，并评估 ID/OOD 泛化。结构与 ``curriculum/linear.py`` 对称，
仅数据生成不同。

关键配置（试出来的经验）
------------------------
- ``data_type='nonlinear'``，``function_callable=lambda x: 2**0.5 * F.relu(x)``，
  ``d_hidden=64``（``d_hidden=d_y`` 时第二层退化为恒等，形式变为 linear + nonlinear_func）。
- ``max_seq_len=200``（非线性用较短序列）。
- 其余与 ``curriculum/linear.py`` 一致：``x_init='zero'``、``init_std='auto'``、
  ``residual_gate=(1,1) fixed``、adam lr=1e-4、关闭 weight_decay、scheduled_training、
  ``save_path='auto'`` / ``load_path='auto'``。

运行：``python experiments/curriculum/nonlinear.py``
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

out_dir = ROOT / 'figures' / 'curriculum'
out_dir.mkdir(parents=True, exist_ok=True)

manual_config = dict(
    data_type='nonlinear',
    function_callable=lambda x: 2**0.5 * F.relu(x),
    max_seq_len=200, d_x=20, seq_len=80, batch_size=64,
    num_blocks=20, num_eff=15, d_model=256, num_heads=8, pe_type=['learned_ape'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    optimizer_type='adam', lr=1e-4,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None,
    epochs=50, steps_per_epoch=200, print_every=5,
    scheduled_training=True,
)

param_groups = [
    {'experiment_name': 'Nonlinear Without Curriculum (Hard Mode)', 'curriculum': {}},
    {'experiment_name': 'Nonlinear With Curriculum (Perfect Path)',
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
                 suptitle='Nonlinear Curriculum Learning Comparison on Looped Transformer')
plt.savefig(out_dir / 'nonlinear_train.png', dpi=120)
plt.close()

# === 阶段 2：加载 checkpoint + ID/OOD 评估 ===
eval_table = ExperimentTable(
    params_groups=[{**p, 'load_path': 'auto'} for p in param_groups],
    manual=manual_config,
)
eval_table.run(result_lists=results_lists_eval, modes=['evaluate'], eval_configs=eval_configs)
eval_table.plot(compare_experiments=False, subplot_shape=(2, -1), suptitle='')
plt.savefig(out_dir / 'nonlinear_eval.png', dpi=120)
plt.close()

print('✅ curriculum/nonlinear 完成，图见 figures/curriculum/nonlinear_{train,eval}.png')
