"""实验：Lorenz 闭环自回归 rollout 的 3D 轨迹可视化（ID + OOD）。

目的
----
用训练好的 Lorenz 模型做**闭环自回归 rollout**：给前 K 步真实轨迹作 prompt，之后让模型
完全用自己的预测作为下一步输入，盲推 1000 步。把"真实流形"与"模型生成的流形"并排画在
3D 相空间中，直观对比模型是否捕捉到了 Lorenz 吸引子的几何与动力学。

关键配置（试出来的经验）
------------------------
- 用 cell 172 配置（num_blocks=20, d_model=256, pe=learned_ape, adam——对齐论文的 lorenz 设定），
  ``load_path='auto'`` 加载 ``curriculum/lorenz.py`` 训练存的 checkpoint。
- rollout：K=150 历史 prompt，rollout_steps=1000 步盲推。
- 滑动窗口更新：每步把模型预测 ``y_pred`` 作为下一步的历史 y（闭环）。
- 两张图：**ID**（默认参数吸引子）与 **OOD**（``rho_shift=5.0``，rho: 28→33 的失真流形）。

前置依赖
--------
需先跑 ``python experiments/curriculum/lorenz.py`` 生成 checkpoint
（``saved_checkpoints/Lorenz_Without_Curriculum_Hard_Mode.pth``）。

运行：``python experiments/lorenz/eval_3d.py``
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (注册 3D 投影)
import numpy as np
import torch

from looped_transformer import ExperimentTable, dataloader

MY_LORENZ_POOL_PATH = 'data/lorenz/length_1000_dt0.01_sigma10.0_beta2.7_rho28.0.pth'

# === 对齐论文的 lorenz 配置（与 curriculum/lorenz.py 一致，以便 load_path='auto' 匹配）===
manual_config = dict(
    data_type='lorenz', d_x=3, d_y=3, max_seq_len=500, seq_len=300, batch_size=64,
    num_blocks=20, num_eff=15, d_model=256, num_heads=8, pe_type=['learned_ape'],
    x_init='zero', init_std='auto', residual_gate=(1, 1), residual_gate_type='fixed',
    lorenz_kwargs=dict(dt=0.01, burn_in=500), load_lorenz_from=MY_LORENZ_POOL_PATH,
    optimizer_type='adam', lr=1e-4,
    layer_weight_decay=1.0, seq_weight_decay=1.0, scheduler_type=None,
    epochs=50, steps_per_epoch=50, print_every=2,
    scheduled_training=True,
)


def rollout_3d(experiment, ood_kwargs, title_true, title_pred, out_path):
    """闭环自回归 rollout 并画 3D 对比图。"""
    model = experiment.model
    model.eval()
    model.to(experiment.device)

    K = 150
    rollout_steps = 1000
    total_needed = K + rollout_steps + 1
    x_spatial, y_spatial = next(dataloader(
        batch_size=1, seq_len=total_needed * 2, data_type='lorenz',
        device=experiment.device, d_x=experiment.d_x, d_y=experiment.d_y,
        ood_kwargs=ood_kwargs,
    ))

    cur_x = x_spatial[:, :K, :].clone()
    cur_y = y_spatial[:, :K, :].clone()
    traj_true, traj_pred = [], []

    for step in range(rollout_steps):
        next_true_x = x_spatial[:, K + step, :].unsqueeze(1)
        next_true_y = y_spatial[:, K + step, :].unsqueeze(1)
        inp_x = torch.cat([cur_x[:, 1:, :], next_true_x], dim=1)
        inp_y = torch.cat([cur_y[:, 1:, :], torch.zeros_like(next_true_y)], dim=1)
        with torch.no_grad():
            y_pred = model(inp_x, inp_y, num_eff=experiment.num_blocks,
                           current_blocks=experiment.num_blocks, is_eval=True)
        traj_true.append(next_true_y.squeeze(0).squeeze(0).cpu().numpy())
        traj_pred.append(y_pred.squeeze(0).cpu().numpy())
        # 闭环：把模型预测喂给下一步作历史
        cur_x = torch.cat([cur_x[:, 1:, :], next_true_x], dim=1)
        cur_y = torch.cat([cur_y[:, 1:, :], y_pred.unsqueeze(1)], dim=1)

    traj_true_np = np.array(traj_true)
    traj_pred_np = np.array(traj_pred)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj_true_np[:, 0], traj_true_np[:, 1], traj_true_np[:, 2], color='#377EB8', lw=1.5)
    ax1.set_title(title_true, fontsize=11, fontweight='bold')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(traj_pred_np[:, 0], traj_pred_np[:, 1], traj_pred_np[:, 2], color='#E41A1C', lw=1.5)
    ax2.set_title(title_pred, fontsize=11, fontweight='bold')
    for ax in [ax1, ax2]:
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'  ✓ {out_path.name}')

    model.to('cpu')  # 释放回 CPU


def main():
    # 加载已训练的 checkpoint（不 run train）
    table = ExperimentTable(
        params_groups=[{
            'experiment_name': 'Lorenz Without Curriculum (Hard Mode)',
            'curriculum': {},
            'load_path': 'auto',
        }],
        manual=manual_config,
    )
    experiment = table.experiments[0]

    out_dir = ROOT / 'figures' / 'lorenz'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ID rollout（默认参数吸引子）
    rollout_3d(experiment, ood_kwargs={},
               title_true='Ground Truth ID Orbit',
               title_pred='Looped Transformer Autoregressive Rollout',
               out_path=out_dir / 'rollout_id.png')

    # OOD rollout（rho 偏移 +5）
    rollout_3d(experiment, ood_kwargs={'rho_shift': 5.0},
               title_true='Ground Truth OOD Orbit ($\\rho=33$)',
               title_pred='Looped Transformer Autoregressive Rollout',
               out_path=out_dir / 'rollout_ood.png')

    print('✅ lorenz/eval_3d 完成，图见 figures/lorenz/rollout_{id,ood}.png')


if __name__ == '__main__':
    main()
