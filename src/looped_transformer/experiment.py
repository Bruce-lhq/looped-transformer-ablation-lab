"""实验管理模块。

提供单实验驾驶舱 LoopedTransformerExperiment，负责训练循环、评估、
结果收集和显存管理。同时包含跨平台的显存监控函数 print_vram_usage。
"""

import time
import torch

from .regression import RegressionSolver
from .data import dataloader


class LoopedTransformerExperiment:
    """单实验驾驶舱。

    封装了模型构建、训练循环、评估、结果收集和显存管理的完整生命周期。
    自动检测并使用可用设备（MPS > CUDA > CPU）。
    支持门控参数与主体参数的分组学习率。

    Attributes:
        model (RegressionSolver): 端到端回归模型。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 运行设备。
        loss_history (list[float]): 每个 epoch 的训练损失。
        residual_gate_history (list[np.ndarray]): 每个 epoch 的门控参数。
    """

    def __init__(self, num_blocks, num_heads=8, d_model=256, lr=1e-4, gate_lr_ratio=100,
                 max_seq_len=4096, d_x=20, d_y=1,
                 seed=None, experiment_name='Experiment',
                 norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape',
                 b_rope_or_upe=10000, head_ratio_upe=2,
                 optimizer_type='adam', wd_adamw=0.01,
                 loop=True, loss_type='mse', bias=False, init_scale=None,
                 residual_gate=(1, 1), residual_gate_type='fixed', residual_random=(1, 0.1),
                 task='regression', timing=True):
        """初始化实验。

        自动检测设备，构建模型和优化器（门控参数使用 gate_lr_ratio 倍的学习率）。

        Args:
            num_blocks (int): Looped Transformer 总迭代层数。
            num_heads (int): 注意力头数，默认 8。
            d_model (int): 模型维度，默认 256。
            lr (float): 基础学习率，默认 1e-4。
            gate_lr_ratio (float): 门控参数学习率 = lr * gate_lr_ratio，默认 100。
            max_seq_len (int): 最大序列长度，默认 4096。
            d_x (int): 输入特征维度，默认 20。
            d_y (int): 输出标签维度，默认 1。
            seed (int or None): 随机种子，None 表示不固定。
            experiment_name (str or None): 实验名称，None 抑制日志输出。
            norm_type (str): 归一化类型，'layernorm' 或 'rmsnorm'。
            ffn_type (str): FFN 类型，'gelu' 或 'swiglu'。
            pe_type (str or list[str]): 位置编码类型。
            b_rope_or_upe (int): RoPE/MS_UPE 基频。
            head_ratio_upe (int): MS_UPE 头倍率。
            optimizer_type (str): 优化器类型，'adam'、'sgd' 或 'adamw'。
            wd_adamw (float): AdamW 优化器的权重衰减系数。
            loop (bool): 是否权重共享。
            loss_type (str): 损失函数，'mse' 或 'l1'。
            bias (bool): RegressionHead 是否使用偏置。
            init_scale (float or None): RegressionHead 初始化标准差。
            residual_gate (tuple or str): 残差门控初始值。
            residual_gate_type (str): 门控类型。
            residual_random (tuple): 随机初始化的 (mean, std)。
            task (str): 任务类型，目前仅支持 'regression'。
            timing (bool): 是否打印初始化耗时。
        """
        if timing:
            start_time = time.time()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.is_offloaded = False
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
            self.generator.manual_seed(seed)
        self.task = task
        if task == 'regression':
            self.model = RegressionSolver(
                num_blocks, num_heads, d_model, d_x=d_x, d_y=d_y, max_seq_len=max_seq_len,
                norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type,
                b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe,
                loop=loop, loss_type=loss_type,
                bias=bias, init_scale=init_scale,
                residual_gate=residual_gate, residual_gate_type=residual_gate_type,
                residual_random=residual_random,
            ).to(self.device)
        gate_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if 'residual_gate' in name:
                gate_params.append(param)
            else:
                base_params.append(param)
        param_groups = [
            {'params': base_params},
            {'params': gate_params, 'lr': lr * gate_lr_ratio},
        ]
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(param_groups, lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(param_groups, lr=lr)
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd_adamw)
        self.d_x = d_x
        self.d_y = d_y
        self.num_blocks = num_blocks
        self.init_time = None
        if experiment_name is not None:
            if timing:
                self.init_time = time.time() - start_time
                print(f"{experiment_name} initialized in {self.init_time:.2f} seconds. Using device: {self.device}")
            else:
                print(f"{experiment_name} initialized. Using device: {self.device}")

    def train(self, batch_size=64, seq_len=80, epochs=20, data_type='linear',
              num_eff=15, sink_padding=None, scheduled_training=True, scheduler_type=None,
              lr_scale=0.1, step_size_scheduler=10,
              print_every=None, timing=False):
        """执行训练循环。

        支持渐进式增加有效层数（scheduled_training）和余弦/StepLR 学习率调度。

        Args:
            batch_size (int): 批次大小，默认 64。
            seq_len (int): 序列长度，默认 80。
            epochs (int): 训练轮数，默认 20。
            data_type (str): 数据类型，默认 'linear'。
            num_eff (int): 有效层数 T，实际取 min(num_eff, num_blocks)。
            sink_padding (int or None): sink token 组数。
            scheduled_training (bool): True 时 current_blocks 从 num_eff 增长到 num_blocks。
            scheduler_type (str or None): LR 调度器，'cosine'、'step' 或 None。
            lr_scale (float): LR 调度器的缩放因子（cosine的eta_min / StepLR的gamma）。
            step_size_scheduler (int): StepLR 调度器的步长。
            print_every (int or None): 每隔几个 epoch 打印一次日志。
            timing (bool): 是否打印训练耗时。
        """
        self.model.train()
        self.max_torch_vram = 0
        self.max_metal_vram = 0
        self.train_time = None
        self.data_loader = dataloader(
            batch_size, seq_len, self.d_x, self.d_y, device=self.device,
            data_type=data_type, sink_padding=sink_padding, generator=self.generator,
        )
        self.loss_history = []
        self.y_pred_norm_history = []
        self.y_true_norm_history = []
        self.residual_gate_history = []
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs,
                eta_min=lr_scale * self.optimizer.param_groups[0]['lr'],
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size_scheduler, gamma=lr_scale,
            )
        num_eff = min(num_eff, self.num_blocks)
        if timing:
            start_time = time.time()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            x_data, y_data = next(self.data_loader)
            if scheduled_training:
                current_blocks = min(num_eff + epoch, self.num_blocks)
            else:
                current_blocks = self.num_blocks
            loss, y_pred_norm, y_true_norm = self.model(
                x_data, y_data, num_eff=num_eff, current_blocks=current_blocks,
                is_eval=False, sink_padding=sink_padding,
            )
            loss = loss.mean()
            loss.backward()
            if torch.backends.mps.is_available():
                self.max_metal_vram = max(self.max_metal_vram, torch.mps.driver_allocated_memory() / (1024 ** 3))
                self.max_torch_vram = max(self.max_torch_vram, torch.mps.current_allocated_memory() / (1024 ** 3))
            self.optimizer.step()
            if scheduler_type is not None:
                scheduler.step()
            self.y_pred_norm_history.append(y_pred_norm)
            self.y_true_norm_history.append(y_true_norm)
            self.loss_history.append(loss.item())
            self.residual_gate_history.append(
                self.model.toy_model.residual_gate.detach().cpu().numpy()
                if hasattr(self.model.toy_model, 'residual_gate') else None
            )
            if print_every is not None and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, current_blocks: {current_blocks}")
        if timing:
            self.train_time = time.time() - start_time
            print(f"Training completed in {self.train_time:.2f} seconds.")
        self.residual_gate_values = self.residual_gate_history[-1]
        self.final_loss = self.loss_history[-1]

    def evaluate(self, batch_size=64, seq_len=80, data_type='linear', loss_type='mse', sink_padding=1):
        """评估模型性能。

        使用 seq_len+1 长度的数据，取最后一层的预测值与真实值计算损失。

        Args:
            batch_size (int): 批次大小。
            seq_len (int): 评估序列长度。
            data_type (str): 数据类型。
            loss_type (str): 评估损失类型，'mse' 或 'l1'。
            sink_padding (int): 评估时的 sink padding。

        Returns:
            float: 评估损失值。
        """
        self.model.eval()
        with torch.no_grad():
            x_eval, y_eval = next(dataloader(
                batch_size, seq_len + 1, self.d_x, self.d_y, device=self.device,
                data_type=data_type, sink_padding=sink_padding, generator=self.generator,
            ))
            y_pred = self.model(
                x_eval, y_eval, num_eff=self.num_blocks,
                current_blocks=self.num_blocks, is_eval=True, sink_padding=sink_padding,
            )
            if loss_type == 'mse':
                mse = torch.nn.functional.mse_loss(y_pred, y_eval[:, -1, :])
                print(f"MSELoss: {mse:.4f}")
                return mse.item()
            elif loss_type == 'l1':
                l1 = torch.nn.functional.l1_loss(y_pred, y_eval[:, -1, :])
                print(f"L1Loss: {l1:.4f}")
                return l1.item()

    def get_results(self, result_list: list[str]):
        """提取指定指标的实验数据。

        根据门控是标量还是向量，自动计算额外的统计量（均值、标准差、相对变化等）。

        Args:
            result_list (list[str]): 需要提取的指标名列表。
                支持的指标见 README.md 的"可用指标一览"。

        Returns:
            dict: {指标名: 数据} 的字典。请求但不可用的 key 会打印 Warning 并被忽略。
        """
        all_results = {
            'init_time': self.init_time,
            'train_time': self.train_time,
            'loss_history': self.loss_history,
            'y_pred_norm_history': self.y_pred_norm_history,
            'y_true_norm_history': self.y_true_norm_history,
            'y_norm_error_history': [pred - true for pred, true in zip(self.y_pred_norm_history, self.y_true_norm_history)],
            'final_loss': self.final_loss,
            'final_y_pred_norm': self.y_pred_norm_history[-1] if self.y_pred_norm_history else None,
            'final_y_true_norm': self.y_true_norm_history[-1] if self.y_true_norm_history else None,
            'residual_gate_history': self.residual_gate_history,
            'final_residual_gate': self.residual_gate_values,
            'captured_attention': self.model.toy_model.captured_attention,
        }
        if any("residual_gate" in key for key in result_list) and self.residual_gate_values is not None:
            if len(self.residual_gate_values.shape) == 1:
                all_results['residual_gate_history_a'] = [gate[0] for gate in self.residual_gate_history]
                all_results['residual_gate_history_b'] = [gate[1] for gate in self.residual_gate_history]
                all_results['residual_gate_history_a_relative'] = [
                    gate[0] - self.residual_gate_history[0][0] for gate in self.residual_gate_history
                ]
                all_results['residual_gate_history_b_relative'] = [
                    gate[1] - self.residual_gate_history[0][1] for gate in self.residual_gate_history
                ]
                # 标量门控的 _mean 就是自身，便于与向量门控混合绘图
                all_results['residual_gate_history_a_mean'] = all_results['residual_gate_history_a']
                all_results['residual_gate_history_b_mean'] = all_results['residual_gate_history_b']
                all_results['residual_gate_history_a_mean_relative'] = all_results['residual_gate_history_a_relative']
                all_results['residual_gate_history_b_mean_relative'] = all_results['residual_gate_history_b_relative']
                all_results['final_residual_gate_a'] = self.residual_gate_values[0]
                all_results['final_residual_gate_b'] = self.residual_gate_values[1]
                all_results['final_residual_gate_a_mean'] = all_results['final_residual_gate_a']
                all_results['final_residual_gate_b_mean'] = all_results['final_residual_gate_b']
            elif len(self.residual_gate_values.shape) == 2:
                all_results['residual_gate_history_a_mean'] = [gate[:, 0].mean() for gate in self.residual_gate_history]
                all_results['residual_gate_history_b_mean'] = [gate[:, 1].mean() for gate in self.residual_gate_history]
                all_results['residual_gate_history_a_mean_relative'] = [
                    gate[:, 0].mean() - self.residual_gate_history[0][:, 0].mean()
                    for gate in self.residual_gate_history
                ]
                all_results['residual_gate_history_b_mean_relative'] = [
                    gate[:, 1].mean() - self.residual_gate_history[0][:, 1].mean()
                    for gate in self.residual_gate_history
                ]
                all_results['residual_gate_history_a_std'] = [gate[:, 0].std() for gate in self.residual_gate_history]
                all_results['residual_gate_history_b_std'] = [gate[:, 1].std() for gate in self.residual_gate_history]
                all_results['final_residual_gate_a_mean'] = self.residual_gate_values[:, 0].mean()
                all_results['final_residual_gate_b_mean'] = self.residual_gate_values[:, 1].mean()
                all_results['final_residual_gate_a_std'] = self.residual_gate_values[:, 0].std()
                all_results['final_residual_gate_b_std'] = self.residual_gate_values[:, 1].std()
        invalid_keys = [key for key in result_list if key not in all_results]
        if invalid_keys:
            print(f"Warning: The following requested results are not available and will be ignored: {invalid_keys}")
        if result_list is not None:
            return {key: all_results[key] for key in result_list if key in all_results}

    def offload_to_cpu(self):
        """将模型和优化器状态转移到 CPU 以节省 GPU 显存。

        同时设置 is_offloaded = True 标记当前状态。
        """
        self.model.to('cpu')
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        self.is_offloaded = True

    def load_to_device(self):
        """将模型和优化器状态从 CPU 恢复到原始设备。

        与 offload_to_cpu 配对使用。
        """
        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.is_offloaded = False


def print_vram_usage(tag="", peak=None):
    """打印当前设备的显存/内存占用情况。

    自动适配 MPS (Apple Silicon)、CUDA (NVIDIA) 和 CPU 三种后端。

    Args:
        tag (str): 日志标签，用于区分不同阶段的显存打印。
        peak (tuple or None): (torch_peak, metal_peak) 峰值显存，
            传入时显示峰值而非当前值。
    """
    if torch.backends.mps.is_available():
        if peak is not None:
            print(f"[{tag}] 显存占用 -> PyTorch分配: {peak[0]:.2f} GB | Metal驱动分配: {peak[1]:.2f} GB")
        else:
            allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
            driver_alloc = torch.mps.driver_allocated_memory() / (1024 ** 3)
            print(f"[{tag}] 显存占用 -> PyTorch分配: {allocated:.2f} GB | Metal驱动分配: {driver_alloc:.2f} GB")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        peak_cuda = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total - reserved
        print(f"[{tag}] 显存占用 -> 已分配: {allocated:.2f} GB | 峰值: {peak_cuda:.2f} GB | 缓存池: {reserved:.2f} GB | 剩余可用: {free:.2f} GB")
    else:
        print(f"[{tag}] 当前运行在 CPU，占用主板内存。")
