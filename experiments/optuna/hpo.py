"""实验：Lorenz 超参搜索（Optuna 多目标）+ 消融热力图。

目的
----
用 Optuna 在 Lorenz 任务上搜索关键超参的最优组合。**多目标**（minimize loss /
maximize norm_ratio / maximize cosine），得到 Pareto 前沿；再为每个超参画
"优越度热力图"（行=3 个指标，列=参数取值），量化每个超参对每个指标的贡献。

搜索空间
--------
- ``gate_lr_ratio``（log-uniform 10~200）、``duration_ratio``（0.3~0.9）；
- ``optimizer_type``（sgd/adam/adamw）、``pe_type``（5 种）、``scheduler_type``（None/cosine/step）；
- ``layer_weight_decay``（0.5~1.0）。

基准配置（manual，固定）
-----------------------
采用 lorenz 最佳配置（muon_adamw + swiglu + ms_upe&alibi，见 ``lorenz/ablation.py``），
Optuna 在其上搜索上述 6 个超参。

⚠️ 耗时
-------
默认 ``n_trials=20``（可改）。每个 trial = 40 epoch × 20 step 训练。完整搜索很久，
建议先小 ``n_trials`` 探路。

运行：``python experiments/optuna/hpo.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import optuna
from looped_transformer import ExperimentTable

MY_LORENZ_POOL_PATH = 'data/lorenz/length_1000_dt0.01_sigma10.0_beta2.7_rho28.0.pth'

manual_config = dict(
    data_type='lorenz', d_x=3, d_y=3, max_seq_len=500, seq_len=200, batch_size=8,
    num_blocks=10, num_eff=7, d_model=128, num_heads=4, ffn_type='swiglu',
    pe_type=['ms_upe', 'alibi'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    lorenz_kwargs=dict(dt=0.01, burn_in=500), load_lorenz_from=MY_LORENZ_POOL_PATH,
    optimizer_type='muon_adamw', lr=2e-2, lr_muon=7e-4, wd_adamw=2e-3,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None,
    epochs=40, steps_per_epoch=20, print_every=None,
    scheduled_training=True,
)

OUT = ROOT / 'figures' / 'optuna'
OUT.mkdir(parents=True, exist_ok=True)


def objective(trial):
    """多目标：返回 (loss, norm_ratio, cosine)，分别 minimize / maximize / maximize。"""
    gate_lr_ratio = trial.suggest_float('gate_lr_ratio', 10.0, 200.0, log=True)
    duration_ratio = trial.suggest_float('duration_ratio', 0.3, 0.9)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['sgd', 'adam', 'adamw'])
    pe_type = trial.suggest_categorical('pe_type', ['alibi', 'learned_ape', 'rope', 'ms_upe', 'ape'])
    scheduler_type = trial.suggest_categorical('scheduler_type', [None, 'cosine', 'step'])
    layer_weight_decay = trial.suggest_float('layer_weight_decay', 0.5, 1.0)
    param_groups = [{
        'experiment_name': f"Trial_{trial.number}",
        'curriculum': {'seq_len': 20, 'duration_ratio': duration_ratio},
        'gate_lr_ratio': gate_lr_ratio,
        'optimizer_type': optimizer_type,
        'pe_type': [pe_type],
        'scheduler_type': scheduler_type,
        'layer_weight_decay': layer_weight_decay,
    }]
    exp_table = ExperimentTable(params_groups=param_groups, manual=manual_config)
    exp_table.run(result_lists=[
        (['final_loss'], 'experiment'),
        (['y_norm_ratio_history'], 'experiment'),
        (['y_cos_history'], 'experiment'),
        (['final_residual_gate'], 'experiment'),
    ], modes=['train'])
    results = exp_table.results[0]
    loss = results['final_loss']
    ratio = results['y_norm_ratio_history'][-1]
    cos = results['y_cos_history'][-1]
    print(f"Trial {trial.number}: Loss={loss:.4f}, Ratio={ratio:.4f}, Cosine={cos:.4f}")
    return loss, ratio, cos


def plot_ablation_heatmaps(study):
    """为每个超参画"优越度热力图"：颜色=优秀度得分，格子数字=真实指标值。"""
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE'].copy()
    param_cols = [c for c in df_complete.columns if c.startswith('params_')]
    metric_rows = ['OOD Loss', 'OOD Norm Ratio', 'OOD Cosine Similarity']
    print(f"检测到 {len(param_cols)} 种超参数，开始绘制消融热力图...")

    for p_col in param_cols:
        p_name = p_col.replace('params_', '')
        df_temp = df_complete.copy()
        if df_temp[p_col].nunique() > 5:
            df_temp[p_col] = pd.qcut(df_temp[p_col], q=4, duplicates='drop').astype(str)
        grouped = df_temp.groupby(p_col)[['values_0', 'values_1', 'values_2']].mean()
        raw_matrix = grouped.T
        raw_matrix.index = metric_rows

        score_matrix = pd.DataFrame(0.0, index=raw_matrix.index, columns=raw_matrix.columns)
        row_loss = raw_matrix.iloc[0]
        score_matrix.iloc[0] = (row_loss.max() - row_loss) / (row_loss.max() - row_loss.min()) if row_loss.max() != row_loss.min() else 1.0
        row_ratio = raw_matrix.iloc[1]
        ratio_dist = np.abs(1.0 - row_ratio)
        score_matrix.iloc[1] = (ratio_dist.max() - ratio_dist) / (ratio_dist.max() - ratio_dist.min()) if ratio_dist.max() != ratio_dist.min() else 1.0
        row_cos = raw_matrix.iloc[2]
        score_matrix.iloc[2] = (row_cos - row_cos.min()) / (row_cos.max() - row_cos.min()) if row_cos.max() != row_cos.min() else 1.0

        plt.figure(figsize=(9, 4.5))
        sns.heatmap(
            data=score_matrix, annot=raw_matrix.values, fmt='.4f', cmap='coolwarm',
            linewidths=2, edgecolor='white',
            cbar_kws={'label': 'Optimization Excellence Score (0=Worst, 1=Best)'},
            xticklabels=raw_matrix.columns, yticklabels=raw_matrix.index,
        )
        plt.title(f"Ablation: Parameter [{p_name}]", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel(f"Values / Binned Intervals of [{p_name}]", fontsize=11, labelpad=8)
        plt.ylabel('Evaluation Outcomes', fontsize=11, labelpad=8)
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(OUT / f'ablation_heatmap_{p_name}.png', dpi=120)
        plt.close()
        print(f"  ✓ [{p_name}] 热力图已保存")


def main(n_trials=20):
    study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'])
    study.optimize(objective, n_trials=n_trials)

    # 平行坐标图（以 cosine 为 target）
    fig = optuna.visualization.plot_parallel_coordinate(study, target=lambda t: t.values[2], target_name='Cosine Similarity')
    try:
        fig.write_image(str(OUT / 'parallel_coordinate.png'))
    except Exception:
        fig.write_html(str(OUT / 'parallel_coordinate.html'))
        print('  (parallel_coordinate 存为 HTML——装 kaleido 可存 PNG)')

    plot_ablation_heatmaps(study)
    print(f'✅ optuna HPO 完成，图见 {OUT}/')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_trials', type=int, default=20, help='Optuna trial 数（默认 20）')
    args = parser.parse_args()
    main(n_trials=args.n_trials)
