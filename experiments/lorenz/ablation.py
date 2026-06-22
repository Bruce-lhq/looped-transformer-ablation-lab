"""实验：Lorenz 系统的穷尽式机制消融。

目的
----
在 Lorenz 下一帧预测任务上，对模型/训练的各个维度做消融对比，找出最佳配置并量化每个
维度对 ID/OOD 性能的贡献。涵盖：PE / optimizer / scheduled_training / scheduler /
num_heads / ffn / loop / norm 共 8 组。

最佳配置（经 optuna 调优得到，作为各消融的 manual 基准）
--------------------------------------------------------
- ``num_blocks=10, num_eff=7, d_model=128, num_heads=4, ffn_type='swiglu'``；
- ``pe_type=['ms_upe', 'alibi']``（PE 叠加在此任务有效，与线性任务结论相反——重要经验）；
- ``optimizer_type='muon_adamw', lr=2e-2, lr_muon=7e-4, wd_adamw=2e-3``；
- ``seq_len=200, batch_size=8, epochs=40, steps_per_epoch=20``。

评估（3 场景，双 Y 轴：ID+Seq 左轴，Param_Shift 右轴）
----------------------------------------------------
- ID Baseline / OOD Param Shift (rho_shift=5) / OOD Seq Extrapolation (×1.2)。

⚠️ 耗时
-------
8 组消融、每组多实验 × (40 epoch 训练 + 3 场景评估)，全程很长。可按需在 ``main`` 里
注释掉不关心的子实验。

运行：``python experiments/lorenz/ablation.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from looped_transformer import ExperimentTable

MY_LORENZ_POOL_PATH = 'data/lorenz/length_1000_dt0.01_sigma10.0_beta2.7_rho28.0.pth'

# === 最佳配置（各消融的 manual 基准）===
manual_config = dict(
    data_type='lorenz', d_x=3, d_y=3, max_seq_len=500, seq_len=200, batch_size=8,
    num_blocks=10, num_eff=7, d_model=128, num_heads=4, ffn_type='swiglu',
    pe_type=['ms_upe', 'alibi'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    lorenz_kwargs=dict(dt=0.01, burn_in=500), load_lorenz_from=MY_LORENZ_POOL_PATH,
    optimizer_type='muon_adamw', lr=2e-2, lr_muon=7e-4, wd_adamw=2e-3,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None,
    epochs=40, steps_per_epoch=20, print_every=None,
    scheduled_training=True, save_path=None, load_path=None,
)

eval_configs = [
    {'eval_name': '1_ID_Baseline',           'ood_kwargs': {}},
    {'eval_name': '2_OOD_Param_Shift',       'ood_kwargs': {'rho_shift': 5.0}},
    {'eval_name': '3_OOD_Seq_Extrapolation', 'ood_kwargs': {'seq_len_scale': 1.2}},
]
# 双 Y 轴：ID + Seq 绑左轴，Param_Shift 绑右轴
result_lists_eval = [
    (['1_ID_Baseline_loss', '3_OOD_Seq_Extrapolation_loss', '|', '2_OOD_Param_Shift_loss'], 'experiment'),
    (['1_ID_Baseline_y_norm_ratio_abs', '3_OOD_Seq_Extrapolation_y_norm_ratio_abs', '|', '2_OOD_Param_Shift_y_norm_ratio_abs'], 'experiment'),
    (['1_ID_Baseline_y_cos_1', '3_OOD_Seq_Extrapolation_y_cos_1', '|', '2_OOD_Param_Shift_y_cos_1'], 'experiment'),
]
train_lists = [
    (['loss_history'], 'epoch'),
    (['y_norm_ratio_history'], 'epoch'),
    (['y_cos_history'], 'epoch'),
]

OUT = ROOT / 'figures' / 'lorenz_ablation'
OUT.mkdir(parents=True, exist_ok=True)


def _ablate(name, params_groups, train_subplot=(-1, 1), eval_subplot=(-1, 2), baseline_idx=None):
    """跑一组消融：train + plot + evaluate + plot + savefig。"""
    table = ExperimentTable(params_groups=params_groups, manual=manual_config)
    table.run(result_lists=train_lists)
    table.plot(suptitle=None, subplot_shape=train_subplot)
    plt.savefig(OUT / f'{name}_train.png', dpi=120)
    plt.close()

    eval_lists = [list(r) for r in result_lists_eval]
    if baseline_idx is not None:
        eval_lists.append((['1_ID_Baseline_loss', '3_OOD_Seq_Extrapolation_loss', '|', '2_OOD_Param_Shift_loss'], 'experiment', baseline_idx))
    table.run(result_lists=eval_lists, modes=['evaluate'], eval_configs=eval_configs, parallel_workers=1)
    table.plot(compare_experiments=True, subplot_shape=eval_subplot, suptitle=None)
    plt.savefig(OUT / f'{name}_eval.png', dpi=120)
    plt.close()
    print(f'  ✓ {name} 完成')


def main():
    # 1) PE 消融（10 种，含叠加组合）
    _ablate('pe', [
        {'experiment_name': 'rope',                'pe_type': ['rope']},
        {'experiment_name': 'learned_ape',         'pe_type': ['learned_ape']},
        {'experiment_name': 'ape',                 'pe_type': ['ape']},
        {'experiment_name': 'alibi',               'pe_type': ['alibi']},
        {'experiment_name': 'ms_upe',              'pe_type': ['ms_upe']},
        {'experiment_name': 'ms_upe_and_alibi',    'pe_type': ['ms_upe', 'alibi']},
        {'experiment_name': 'learned_ape_and_alibi', 'pe_type': ['learned_ape', 'alibi']},
        {'experiment_name': 'rope_and_alibi',      'pe_type': ['rope', 'alibi']},
        {'experiment_name': 'ape_and_alibi',       'pe_type': ['ape', 'alibi']},
        {'experiment_name': 'ms_upe_and_learned_ape', 'pe_type': ['ms_upe', 'learned_ape']},
    ], train_subplot=(1, -1), baseline_idx=3)  # 以 alibili 为基线

    # 2) Optimizer 消融（adamw / muon_adamw / nora_adamw）
    _ablate('optimizer', [
        {'experiment_name': 'adamw_5e-3_wd_5e-3',      'optimizer_type': 'adamw',      'lr': 5e-3, 'wd_adamw': 5e-3},
        {'experiment_name': 'adamw_2e-3_wd_8e-3',      'optimizer_type': 'adamw',      'lr': 2e-3, 'wd_adamw': 8e-3},
        {'experiment_name': 'muon_7e-4_adamw_2e-2_wd_2e-3', 'optimizer_type': 'muon_adamw', 'lr': 2e-2, 'lr_muon': 7e-4, 'wd_adamw': 2e-3},
        {'experiment_name': 'muon_1e-3_adamw_1e-2_wd_5e-3', 'optimizer_type': 'muon_adamw', 'lr': 1e-2, 'lr_muon': 1e-3, 'wd_adamw': 5e-3},
        {'experiment_name': 'nora_1e-4_adamw_2e-2_wd_5e-3', 'optimizer_type': 'nora_adamw',  'lr': 2e-2, 'lr_nora': 1e-4, 'wd_adamw': 5e-3},
        {'experiment_name': 'nora_1e-3_adamw_2e-2_wd_2e-3', 'optimizer_type': 'nora_adamw',  'lr': 2e-2, 'lr_nora': 1e-3, 'wd_adamw': 2e-3},
    ], train_subplot=(1, -1), baseline_idx=2)

    # 3) Scheduled Training（kick-start 渐进层数）
    _ablate('scheduled_training', [
        {'experiment_name': 'on',  'scheduled_training': True},
        {'experiment_name': 'off', 'scheduled_training': False},
    ], eval_subplot=(-1, 1))

    # 4) Scheduler 消融（off / cosine / step）
    _ablate('scheduler', [
        {'experiment_name': 'off',         'scheduler_type': None},
        {'experiment_name': 'cosine_1e-5', 'scheduler_type': 'cosine', 'eta_min': 1e-5},
        {'experiment_name': 'cosine_1e-4', 'scheduler_type': 'cosine', 'eta_min': 1e-4},
        {'experiment_name': 'step_100',    'scheduler_type': 'step', 'step_size_scheduler': 100},
        {'experiment_name': 'step_300',    'scheduler_type': 'step', 'step_size_scheduler': 300},
        {'experiment_name': 'step_600',    'scheduler_type': 'step', 'step_size_scheduler': 600},
    ], baseline_idx=0)

    # 5) num_heads 消融
    _ablate('num_heads', [
        {'experiment_name': '1_head',  'num_heads': 1},
        {'experiment_name': '2_heads', 'num_heads': 2},
        {'experiment_name': '4_heads', 'num_heads': 4},
        {'experiment_name': '8_heads', 'num_heads': 8},
        {'experiment_name': '16_heads', 'num_heads': 16},
    ], train_subplot=(1, -1), baseline_idx=2)  # 4 heads（=最佳配置）为基线

    # 6) FFN 消融（gelu / swiglu）
    _ablate('ffn', [
        {'experiment_name': 'gelu',   'ffn_type': 'gelu'},
        {'experiment_name': 'swiglu', 'ffn_type': 'swiglu'},
    ], eval_subplot=(-1, 1))

    # 7) loop 消融（权重共享 / 独立层）
    _ablate('loop', [
        {'experiment_name': 'loop'},
        {'experiment_name': 'nonloop', 'loop': False, 'num_blocks': 10, 'num_eff': 10},
    ], eval_subplot=(-1, 1))

    # 8) Norm 消融（layernorm / rmsnorm）
    _ablate('norm', [
        {'experiment_name': 'layernorm', 'norm_type': 'layernorm'},
        {'experiment_name': 'rmsnorm',   'norm_type': 'rmsnorm'},
    ], eval_subplot=(-1, 1))

    print(f'✅ lorenz ablation 全部完成，图见 {OUT}/')


if __name__ == '__main__':
    main()
