"""实验：三大任务（linear / nonlinear / lorenz）统一 OOD 分析与散点图。

目的
----
把 linear / nonlinear / lorenz 三个任务、With/Without Curriculum 两种条件、3 种 OOD 场景
共 18 个评估点，统一画在对数流形空间（Norm Ratio R vs Loss L）中，观察是否存在的统一标度律。
左图按 OOD 场景着色，右图按系统 Data Type 着色。

前置依赖
--------
本脚本 ``modes=['evaluate']`` 需要先有训练好的 checkpoint（``load_path='auto'`` 自动匹配
``experiment_name``）。请先依次跑：
    python experiments/curriculum/linear.py
    python experiments/curriculum/nonlinear.py
    python experiments/curriculum/lorenz.py

关键配置
--------
- 三任务共用模型骨架（num_blocks=20, d_model=256, pe=learned_ape, x_init='zero',
  init_std='auto', residual_gate=(1,1) fixed, adam lr=1e-4, 关闭 weight_decay）。
- 评估 3 场景：ID Baseline / OOD Scale(或 Param Shift) / Seq Extrapolation。
- Loss 做了 ``L / y_true_norm²`` 归一化后再进对数空间。

运行：``python experiments/curriculum/unified_analysis.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F
from looped_transformer import ExperimentTable

# === 共用模型骨架 ===
manual = dict(
    batch_size=64, num_blocks=20, num_eff=15, d_model=256, num_heads=8, pe_type=['learned_ape'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    optimizer_type='adam', lr=1e-4,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None, scheduled_training=True,
)
manual_linear = dict(data_type='linear', d_x=20, seq_len=80, epochs=50, steps_per_epoch=200, print_every=1)
manual_nonlinear = dict(
    data_type='nonlinear', function_callable=lambda x: 2**0.5 * F.relu(x),
    max_seq_len=200, d_x=20, seq_len=80, epochs=50, steps_per_epoch=200, print_every=5,
)
MY_LORENZ_POOL_PATH = 'data/lorenz/length_1000_dt0.01_sigma10.0_beta2.7_rho28.0.pth'
manual_lorenz = dict(
    data_type='lorenz', d_x=3, d_y=3, max_seq_len=500, seq_len=300,
    lorenz_kwargs=dict(dt=0.01, burn_in=500), load_lorenz_from=MY_LORENZ_POOL_PATH,
    epochs=50, steps_per_epoch=50, print_every=2,
)

# load_path='auto' 自动匹配 experiment_name → saved_checkpoints/<safe_name>.pth
params_groups = [
    {**manual_linear,    'experiment_name': 'Linear Without Curriculum (Hard Mode)',    'curriculum': {}, 'load_path': 'auto'},
    {**manual_linear,    'experiment_name': 'Linear With Curriculum (Perfect Path)',    'curriculum': {'d_x': 5, 'seq_len': 10, 'duration_ratio': 0.8}, 'load_path': 'auto'},
    {**manual_nonlinear, 'experiment_name': 'Nonlinear Without Curriculum (Hard Mode)', 'curriculum': {}, 'load_path': 'auto'},
    {**manual_nonlinear, 'experiment_name': 'Nonlinear With Curriculum (Perfect Path)', 'curriculum': {'d_x': 5, 'seq_len': 10, 'duration_ratio': 0.8}, 'load_path': 'auto'},
    {**manual_lorenz,    'experiment_name': 'Lorenz Without Curriculum (Hard Mode)',    'curriculum': {}, 'load_path': 'auto'},
    {**manual_lorenz,    'experiment_name': 'Lorenz With Curriculum (Perfect Path)',    'curriculum': {'seq_len': 20, 'duration_ratio': 0.8}, 'load_path': 'auto'},
]

eval_configs = [
    {'eval_name': '1_ID_Baseline',                  'ood_kwargs': {}},
    {'eval_name': '2_OOD_Scale_x2_or_Param_Shift',  'ood_kwargs': {'x_scale': 2.0, 'rho_shift': 5.0}},
    {'eval_name': '3_OOD_Seq_Extrapolation',        'ood_kwargs': {'seq_len_scale': 1.2}},
]
result_lists = [
    (['1_ID_Baseline_sink_scores', '2_OOD_Scale_x2_or_Param_Shift_sink_scores', '3_OOD_Seq_Extrapolation_sink_scores'], 'block'),
    (['1_ID_Baseline_y_norm_ratio', '2_OOD_Scale_x2_or_Param_Shift_y_norm_ratio', '3_OOD_Seq_Extrapolation_y_norm_ratio'], 'experiment'),
    (['1_ID_Baseline_y_cos', '2_OOD_Scale_x2_or_Param_Shift_y_cos', '3_OOD_Seq_Extrapolation_y_cos'], 'experiment'),
    (['1_ID_Baseline_loss', '2_OOD_Scale_x2_or_Param_Shift_loss', '3_OOD_Seq_Extrapolation_loss'], 'experiment'),
]


def plot_unified_dots(target_table):
    """把所有任务×所有场景的评估点画在统一的 log-log 流形空间（R vs L）。

    一式两份：左图按 OOD 场景着色，右图按 Data Type 着色。
    Loss 做了 L/(y_true_norm²) 归一化以跨任务可比。
    """
    all_points = []
    eval_names = ['1_ID_Baseline', '2_OOD_Scale_x2_or_Param_Shift', '3_OOD_Seq_Extrapolation']
    for i, exp in enumerate(target_table.experiments):
        res = exp.eval_results
        data_type = target_table.train_parameters[i].get('data_type', 'unknown')
        for eval_name in eval_names:
            loss_key = f"{eval_name}_loss"
            ratio_key = f"{eval_name}_y_norm_ratio"
            y_true_key = f"{eval_name}_y_true_norm"
            if loss_key in res and ratio_key in res and y_true_key in res:
                L_val = res[loss_key]
                L_normalized = L_val / (res[y_true_key] ** 2 + 1e-8)
                R_val = res[ratio_key]
                if (L_normalized is not None and R_val is not None and
                        not np.isnan(L_normalized) and not np.isnan(R_val) and
                        L_normalized > 0 and R_val > 1e-5):
                    all_points.append({'R': R_val, 'L': L_normalized, 'OOD': eval_name, 'DataType': data_type})
    if len(all_points) < 2:
        print("❌ 有效数据点不足，请确认前置训练脚本（curriculum/{linear,nonlinear,lorenz}.py）已跑完并存好 checkpoint。")
        return
    print(f"[全局散点图] 激活样本量: {len(all_points)} 个点")

    ryb = {'red': '#E41A1C', 'yellow': '#FFCC00', 'blue': '#377EB8'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ood_map = {
        '1_ID_Baseline':                  ('1_ID_Baseline', ryb['blue']),
        '2_OOD_Scale_x2_or_Param_Shift':  ('2_OOD_Scale_x2_or_Param_Shift', ryb['yellow']),
        '3_OOD_Seq_Extrapolation':        ('3_OOD_Seq_Extrapolation', ryb['red']),
    }
    for ood_cat, (label, color) in ood_map.items():
        pts = [p for p in all_points if p['OOD'] == ood_cat]
        if pts:
            ax1.scatter([p['R'] for p in pts], [p['L'] for p in pts], color=color, label=label, s=120, edgecolor='k', alpha=0.9, zorder=3)
    ax1.set_title('Color Coded by OOD Evaluation Type', fontsize=12, fontweight='bold', pad=10)

    dtype_map = {
        'linear':    ('LINEAR', ryb['blue']),
        'nonlinear': ('NONLINEAR', ryb['yellow']),
        'lorenz':    ('LORENZ', ryb['red']),
    }
    for dtype_cat, (label, color) in dtype_map.items():
        pts = [p for p in all_points if p['DataType'] == dtype_cat]
        if pts:
            ax2.scatter([p['R'] for p in pts], [p['L'] for p in pts], color=color, label=label, s=120, edgecolor='k', alpha=0.9, zorder=3)
    ax2.set_title('Color Coded by System Data Type', fontsize=12, fontweight='bold', pad=10)

    for ax in [ax1, ax2]:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Norm Ratio $R$ ($||\\hat{y}|| / ||y||$)', fontsize=11)
        ax.set_ylabel('Loss $L$', fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.4, color='gray')
        ax.legend(fontsize=9, loc='best', frameon=True, facecolor='white', edgecolor='gainsboro')
    plt.suptitle("Unified Dots: 18 Checkpoints Log-Log Mapping", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()


# === 运行统一评估 ===
unified_table = ExperimentTable(params_groups=params_groups, manual=manual)
unified_table.run(result_lists=result_lists, modes=['evaluate'], eval_configs=eval_configs, parallel_workers=1)
unified_table.plot(compare_experiments=False, subplot_shape=(3, -1), figure_size=(18, 12), suptitle='')

out_dir = ROOT / 'figures' / 'curriculum'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'unified_eval.png', dpi=120)
plt.close()

# === 散点图 ===
plot_unified_dots(unified_table)
plt.savefig(out_dir / 'unified_dots.png', dpi=120)
plt.close()

print('✅ unified_analysis 完成，图见 figures/curriculum/unified_{eval,dots}.png')
