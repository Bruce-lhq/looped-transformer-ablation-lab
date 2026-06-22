# Looped Transformer：机制消融与动力学分析

对 **Looped Transformer**（所有层共享权重、循环迭代的 Transformer 变体）的穷尽式机制消融
与动力学分析框架。下游任务覆盖**线性回归、非线性回归、Lorenz 混沌系统下一帧预测**三大场景，
支持 curriculum learning、Muon/Nora 混合优化器、OOD 泛化评估、checkpoint 断点续训。

> 由原 `Looped_Transformer.ipynb` 拆解而来，以可维护的小 `.py` 文件为主，便于增量修改。

## 目录

1. [理论](#理论)
2. [模块拼装](#模块拼装)
3. [目录结构](#目录结构)
4. [快速启动](#快速启动)
5. [实验框架：ExperimentTable](#实验框架experimenttable)
6. [实验脚本](#实验脚本)
7. [依赖](#依赖)
8. [待完成与未来工作](#待完成与未来工作)

---

## 理论

### 循环迭代与残差门控

所有层共享同一组权重 $\theta$，每层的输入是前一层输出与初始输入的加权组合：

$$
\begin{aligned}
h_0 &= \text{input} \\
h_l &= \text{TransformerBlock}(a \cdot h_{l-1} + b \cdot h_0 \mid \theta) \quad l=1,2,\dots,L
\end{aligned}
$$

$(a, b)$ 为**残差门控**（`residual_gate`），控制当前状态与原始输入之间的信息流，可为固定值、
可学习标量或可学习逐维向量。

### 截断损失（防梯度爆炸）

只让最后 $T$ 层参与损失计算与梯度回传，前 $L-T$ 层在边界处 `detach()` 释放计算图：

$$\text{Loss}(\theta)=\mathbb{E}\left[ \frac{1}{L-b_0} \sum_{t=b_0}^{L} \frac{1}{k+1} \sum_{i=0}^{k} (Y_t(P^i \mid \theta) - f(x_{i+1}))^2 \right]$$

- $T$ 即有效层数 `num_eff`，$b_0 = \max(L - T, 0)$；
- 可选的 `layer_weight_decay` / `seq_weight_decay` 对不同层 / 不同序列位置做指数加权。

### Curriculum Learning（打破"d_x 之墙"）

高维线性回归下 loss 极难下降（"d_x 之墙"）。Curriculum 在训练前 `duration_ratio` 比例的步数内，
从低维短序列起步、线性放大到目标 `(d_x, seq_len)`，有效打破这堵墙：

```python
curriculum = {'d_x': 5, 'seq_len': 10, 'duration_ratio': 0.8}
```

> 注意：Lorenz 的 `d_x=3` 是物理维度不能变，curriculum 只演进 `seq_len`。

### 下游任务

| 任务 | 数据生成 | 维度 |
|---|---|---|
| 线性回归 | $y = x^\top w$，每 batch 采样 $w \sim \mathcal{N}(0, I/d_x)$ | $d_x=20, d_y=1$ |
| 非线性回归 | $y = w_2 \cdot \sigma(w_1 \cdot x)$，$\sigma$ 可换 | $d_x=20, d_y=1, d_{hidden}=64$ |
| Lorenz | RK4 积分 Lorenz 方程，下一帧状态预测 | $d_x=3, d_y=3$ |

Prompt 统一为交织序列 $(x_1, y_1, \dots, x_k, y_k, x_{test})$，取最后一层输出的倒数第二 token 作预测。
GPT-2 风格初始化（`init_std`）+ `ln_f` 兜底归一化 + 残差输出层缩放（`std_res = init_std / sqrt(2L)`）。

---

## 模块拼装

```
Looped Transformer 实验台
├── core/                             # 🟢 写定的底层积木（稳定层）
│   ├── position_encoding.py          # APE / LearnedAPE / ALiBi / RoPE / MS_UPE（自创多尺度解绑）
│   ├── swiglu.py                     # SwiGLU（与 nn.GELU 并列的 FFN 激活）
│   ├── probes.py                     # AttentionProbe（捕获注意力矩阵）/ SinkMetricsProbe（sink score/rate）
│   ├── lorenz.py                     # Lorenz 底层动力学：导数 / RK4 / kernel / 离线池化
│   ├── optimizers.py                 # HybridOptimizer（多优化器包装）+ Nora（正交化优化器）
│   └── print_vram_usage.py           # 跨平台显存监控（MPS/CUDA/CPU）
├── attention.py                      # MultiHeadAttention（PE 分发 + 训练 fused / 推理捕获）
├── transformer_block.py              # Pre-Norm Block（LayerNorm/RMSNorm + MHA + GELU/SwiGLU）
├── toy_model.py                      # Looped 引擎（权重共享、残差门控、num_eff 截断、x_init、probe 集成）
├── regression.py                     # RegressionHead（双通道投影+拉链交织）/ PredictionLoss（ln_f+加权）/ Solver
├── data_generators.py                # linear / nonlinear / lorenz 三个 *_data_generator + lorenz 缓存
├── dataloader.py                     # 按 data_type 统一分发 + sink padding
├── experiment.py                     # LoopedTransformerExperiment（训练/评估/检查点/结果收集）
├── experiment_table.py               # ExperimentTable（多实验调度 + 自动绑图）
└── default_setup.py                  # default_setup()：集中管理的默认参数字典
```

**包内依赖链**（无环）：`core/`（独立积木）→ `attention` → `transformer_block` → `toy_model` → `regression`
→ `experiment` → `experiment_table`；`data_generators`（用 `core.lorenz`）→ `dataloader` → `experiment`；
`default_setup` → `experiment_table`。

顶层 `__init__.py` 再导出全部公开 API，使用体验扁平：`from looped_transformer import ExperimentTable, ToyModel, ...`。

---

## 目录结构

```
Looped_Transformer/
├── looped_transformer/        # 核心包（见上"模块拼装"）
│   └── core/
├── experiments/               # 实验脚本（每个一个主题，沉淀"怎么配参"的经验）
│   ├── linear/                # 线性回归各项消融
│   ├── nonlinear/             # 非线性回归
│   ├── curriculum/            # 三任务 curriculum + 统一分析
│   ├── lorenz/                # Lorenz 消融 + 3D rollout
│   └── optuna/                # 超参搜索
├── data/lorenz/               # Lorenz 离线轨迹池（gitignored，由脚本生成）
├── saved_checkpoints/         # 训练 checkpoint（gitignored）
├── figures/                   # 实验输出图（gitignored）
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 快速启动

```bash
# 安装依赖（用 uv pip）
uv pip install -r requirements.txt
# 或：pip install -r requirements.txt

# 跑一个最小实验（线性 PE 对比）
python experiments/linear/pe_compare.py
```

设备自动检测 **MPS > CUDA > CPU**。每个实验脚本独立可跑，图存到 `figures/<分类>/`。

最小用法（库代码）：

```python
from looped_transformer import ExperimentTable

table = ExperimentTable(params_groups=[
    {'pe_type': ['alibi'], 'experiment_name': 'ALiBi'},
    {'pe_type': ['rope'],  'experiment_name': 'RoPE'},
])
table.run(result_lists=[(['loss_history'], 'epoch')])
table.plot()
```

---

## 实验框架：ExperimentTable

`ExperimentTable` 是多实验对比的核心调度器。底层逻辑是"全量默认 + 局部覆写"：
`default_setup()` 提供所有默认值，`params_groups` 中只写要改的 key，`manual` 全局覆写。

### 核心流程

1. `__init__(params_groups, manual=None)` — 加载默认参数、逐实验覆写、构造所有实验对象。
2. `run(result_lists, modes=['train'], parallel_workers=1, eval_configs=None)` — 执行。
3. `plot(compare_experiments, subplot_shape, figure_size, ...)` — 渲染对比图。

### result_lists 格式（支持双 Y 轴与多横轴）

```python
result_lists = [
    # 折线：横轴 epoch
    (['loss_history'], 'epoch'),
    # 柱状图：横轴 experiment（metric 为单值）
    (['final_loss', 'final_y_pred_norm', '|', 'final_residual_gate_a'], 'experiment'),
    # 双 Y 轴：'|' 左侧绑左轴，右侧绑右轴
    (['loss_history', '|', 'residual_gate_history_a'], 'epoch'),
    # baseline：第 3 项为基线索引（0-based），画相对差值
    (['loss_history'], 'epoch', 0),
    # block 横轴：仅限 eval 与 captured 的 sink 指标
    (['1_ID_Baseline_sink_scores', '2_OOD_Param_Shift_sink_scores'], 'block'),
]
```

### run 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `result_lists` | `list[tuple]` | 必填 | 见上 |
| `modes` | `list[str]` | `['train']` | `'train'` 和/或 `'evaluate'` |
| `parallel_workers` | `int` | `1` | 并行线程数（>1 多线程压测） |
| `eval_configs` | `list[dict]` | `None` | 评估配置，如 `[{'eval_name':'id','ood_kwargs':{}}, {'eval_name':'ood','ood_kwargs':{'x_scale':2.0}}]`。`eval_name` 作结果 key 前缀 |

### plot 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `compare_experiments` | `True` | True：同指标跨实验横向对比；False：每实验独立子图（`'|'` 双 Y 轴） |
| `subplot_shape` | `(1,-1)` | 子图网格，-1 自动计算 |
| `figure_size` | `None` | None 时取 `(8*cols, 6*rows)` |
| `suptitle` | `'Looped...'` | 顶部标题 |

### 可覆写参数（default_setup）

按模块分组（摘录，完整见 `looped_transformer/default_setup.py`）：

| 模块 | 关键 key |
|---|---|
| MultiHeadAttention | `num_heads`, `d_model`, `max_seq_len`, `pe_type` |
| TransformerBlock | `norm_type`, `ffn_type` |
| ToyModel | `num_blocks`, `loop`, `residual_gate`, `residual_gate_type`, `x_init`, `sink_threshold` |
| RegressionSolver | `d_x`, `d_y`, `init_std`(`'auto'`/float/None), `layer_weight_decay`, `seq_weight_decay` |
| LoopedTransformerExperiment | `lr`, `lr_muon`, `lr_nora`, `gate_lr_ratio`, `optimizer_type`(`'muon_adamw'`/`'nora_adamw'`/...), `seed`, `load_path`(`'auto'`) |
| dataloader | `batch_size`, `seq_len`, `sink_padding`, `d_hidden`, `function_callable`, `lorenz_kwargs`, `load_lorenz_from` |
| train | `epochs`, `steps_per_epoch`, `data_type`, `scheduler_type`, `eta_min`, `scheduled_training`, `curriculum`, `save_path`(`'auto'`) |

### 常见报错

| 报错 | 原因 |
|---|---|
| `ValueError: Unknown parameter: {key}` | params_groups 拼错 key |
| `pe_type` 格式错误 | 必须用列表 `['ape']`，不能是字符串 |
| 请求的指标被静默忽略 | metric 名须与 `get_results()` 返回的 key 完全一致 |

---

## 实验脚本

每个脚本顶部注释块写明**实验目的 + 关键配置 + 为什么这么配**（参数是试出来的）。

| 脚本 | 主题 |
|---|---|
| `experiments/linear/observations.py` | loss 曲线 + 范数观测（含 curriculum 打破 d_x 之墙） |
| `experiments/linear/scheduled_vs_non.py` | Scheduled Training 对比 |
| `experiments/linear/pe_compare.py` | 5 种 PE + 叠加对比（ALiBi 为基线） |
| `experiments/linear/sink_padding.py` | sink padding 消融 |
| `experiments/linear/residual_gate.py` | 残差门控类型/初值消融 + 漂移观测 |
| `experiments/linear/scheduler.py` | None/Cosine/Step 调度器对比 |
| `experiments/nonlinear/regression.py` | 非线性回归训练观测（loss/norm/gate 双 Y 轴） |
| `experiments/curriculum/linear.py` | 线性 curriculum + ID/OOD 评估（训-存-载-评） |
| `experiments/curriculum/nonlinear.py` | 非线性 curriculum + ID/OOD 评估 |
| `experiments/curriculum/lorenz.py` | Lorenz curriculum + ID/Param-Shift/Seq 评估 |
| `experiments/curriculum/unified_analysis.py` | 三任务统一 OOD 分析 + log-log 散点图 |
| `experiments/lorenz/ablation.py` | Lorenz 8 维消融（PE/optim/scheduled/scheduler/heads/ffn/loop/norm） |
| `experiments/lorenz/eval_3d.py` | Lorenz 闭环自回归 1000 步 rollout 的 3D 轨迹（ID + OOD） |
| `experiments/optuna/hpo.py` | Lorenz 多目标超参搜索 + 消融热力图 |

---

## 依赖

| 用途 | 包 |
|---|---|
| 核心训练 | `torch>=2.10`（用 `torch.optim.Muon`）、`numpy`、`matplotlib` |
| 实验脚本 | `optuna`、`pandas`、`seaborn` |

```bash
uv pip install -r requirements.txt
```

> Nora 优化器已迁入 `looped_transformer/core/optimizers.py`，不是 pip 包。
> Lorenz 离线数据池由 `create_lorenz_pool()` 生成（约 2.4GB，gitignored）；缺失时自动 fallback 到实时 RK4。

---

## 待完成与未来工作

- **实验框架**：`plot()` 对 `'block'` 横轴的多场景对比、3D 动画导出；
- **任务扩展**：ODE 数据生成器、逻辑推理任务；
- **模型探针**：各层线性探针检测闭式解 $w=(X^\top X)^{-1}X^\top y$ 的涌现；
- **归一化**：W_Q/W_K/W_V 的谱归一化对稳定性的影响；
- **Grokking**：大 epochs/batch 下观察顿悟现象。
