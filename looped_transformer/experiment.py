"""实验管理模块。

提供单实验驾驶舱 ``LoopedTransformerExperiment``，封装模型构建、训练循环（含
curriculum learning、scheduled training、cosine/step scheduler、checkpoint 保存/加载）、
评估（含 OOD 配置）、结果收集（含 sink 指标、残差门控统计）与显存管理。
自动检测设备 MPS > CUDA > CPU。
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .regression import RegressionSolver
from .dataloader import dataloader
from .core.optimizers import HybridOptimizer, Nora


class LoopedTransformerExperiment:
    """单实验驾驶舱。

    封装模型构建、训练、评估、结果收集与显存管理的完整生命周期。自动检测设备
    （MPS > CUDA > CPU）。支持门控参数与主体参数的分组学习率，以及 Muon/Nora 与
    Adam(W)/SGD 组合的 HybridOptimizer。支持 curriculum learning 与 checkpoint 断点续训。

    Attributes:
        model (RegressionSolver): 端到端回归模型。
        optimizer (torch.optim.Optimizer or HybridOptimizer): 优化器。
        device (torch.device): 运行设备。
        loss_history (list[float]): 每个 epoch 的平均训练损失。
        y_pred_norm_history (list[float]): 每个 epoch 的预测值 RMS 范数。
        y_true_norm_history (list[float]): 每个 epoch 的真实值 RMS 范数。
        residual_gate_history (list): 每个 epoch 的门控参数（numpy）。
        sink_score_history (list[float]): 每个 epoch 的平均 sink score。
        sink_rate_history (list[float]): 每个 epoch 的平均 sink rate。
        eval_results (dict): 评估结果（以 ``{eval_name}_*`` 为 key）。
    """

    def __init__(self, num_blocks, num_heads=8, d_model=256, lr=1e-4, gate_lr_ratio=100, max_seq_len=4096, d_x=20, d_y=1,
                 seed=None, experiment_name='Experiment',
                 norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape', b_rope_or_upe=10000, head_ratio_upe=2,
                 optimizer_type='adam', wd_adamw=0.01,
                 lr_muon=2e-3, lr_nora=5e-4,
                 loop=True, loss_type='mse', bias=False, init_scale=None, init_std=0.02, x_init='prompt', sink_threshold=0.3,
                 residual_gate=(1, 1), residual_gate_type='fixed', residual_random=(1, 0.1),
                 task='regression', timing=True, load_path=None, print_on=True,
                 layer_weight_decay=1.0, seq_weight_decay=1.0):
        """初始化实验：检测设备、构建模型与优化器、可选加载 checkpoint。

        优化器构建逻辑：
        - ``optimizer_type`` 含 'muon'/'nora'：≥2 维参数走 Muon/Nora，其余走 Adam(W)/SGD，
          用 HybridOptimizer 包装；门控参数始终用 ``lr * gate_lr_ratio`` 的学习率。
        - 否则：base 参数与 gate 参数两组，走单一的 Adam/SGD/AdamW。

        Args:
            num_blocks (int): Looped Transformer 总迭代层数。
            num_heads (int): 注意力头数。
            d_model (int): 模型维度。
            lr (float): 基础学习率。
            gate_lr_ratio (float): 门控参数学习率 = lr * gate_lr_ratio。
            max_seq_len (int): 最大序列长度。
            d_x (int): 输入特征维度。
            d_y (int): 输出标签维度。
            seed (int or None): 随机种子，None 不固定。
            experiment_name (str or None): 实验名称，None 抑制日志。
            norm_type (str): 归一化类型，'layernorm' 或 'rmsnorm'。
            ffn_type (str): FFN 类型，'gelu' 或 'swiglu'。
            pe_type (str or list[str]): 位置编码类型。
            b_rope_or_upe (int): RoPE/MS_UPE 基频。
            head_ratio_upe (int): MS_UPE 头倍率。
            optimizer_type (str): 优化器类型，'adam'/'sgd'/'adamw'，或前缀 'muon_'/'nora_'。
            wd_adamw (float): AdamW 权重衰减。
            lr_muon (float): Muon 学习率。
            lr_nora (float): Nora 学习率。
            loop (bool): 是否权重共享。
            loss_type (str): 损失函数，'mse' 或 'l1'。
            bias (bool): RegressionHead 是否使用偏置。
            init_scale (float or None): RegressionHead 独立初始化标准差。
            init_std (float or 'auto' or None): GPT-2 风格全局初始化标准差。
            x_init (str): ToyModel 初始状态来源。
            sink_threshold (float): SinkMetricsProbe 阈值。
            residual_gate (tuple or str): 残差门控初始值。
            residual_gate_type (str): 门控类型。
            residual_random (tuple): 随机门控的 (mean, std)。
            task (str): 任务类型，目前仅 'regression'。
            timing (bool): 是否打印初始化耗时。
            load_path (str or None): 预训练 checkpoint 路径，None 不加载。
            print_on (bool): 是否打印日志。
            layer_weight_decay (float): PredictionLoss 层加权系数。
            seq_weight_decay (float): PredictionLoss 序列位置加权系数。

        Raises:
            ValueError: optimizer_type 既不含 muon/nora 也不属于 adam/sgd/adamw 时。
        """
        if timing:
            start_time = time.time()
        # 设备选择：优先使用 MPS（适用于 Apple Silicon），其次是 CUDA（适用于 NVIDIA GPU），最后退回 CPU
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
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
        # 模型和优化器
        self.task = task
        if task == 'regression':
            self.model = RegressionSolver(num_blocks=num_blocks, num_heads=num_heads, d_model=d_model, d_x=d_x, d_y=d_y, max_seq_len=max_seq_len,
                                          norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type, b_rope_or_upe=b_rope_or_upe,
                                          head_ratio_upe=head_ratio_upe, loop=loop, loss_type=loss_type,
                                          bias=bias, init_scale=init_scale, init_std=init_std, x_init=x_init, sink_threshold=sink_threshold,
                                          residual_gate=residual_gate, residual_gate_type=residual_gate_type, residual_random=residual_random,
                                          layer_weight_decay=layer_weight_decay, seq_weight_decay=seq_weight_decay
                                          ).to(self.device)
        self.eval_results = {}
        gate_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if 'residual_gate' in name:
                gate_params.append(param)
            else:
                base_params.append(param)
        if 'muon' in optimizer_type or 'nora' in optimizer_type:
            spec_params = [param for param in base_params if param.ndim >= 2]
            other_params = [param for param in base_params if param.ndim < 2]
            if 'muon' in optimizer_type:
                opt_spec = torch.optim.Muon(spec_params, lr=lr_muon)
            elif 'nora' in optimizer_type:
                opt_spec = Nora([{'params': spec_params, 'lr': lr_nora, 'is_nora': True}], lr_nora=lr_nora)
            other_param_groups = [{'params': other_params},
                                  {'params': gate_params, 'lr': lr * gate_lr_ratio}]
            if 'adamw' in optimizer_type:
                opt_base = torch.optim.AdamW(other_param_groups, lr=lr, weight_decay=wd_adamw)
            elif 'adam' in optimizer_type:
                opt_base = torch.optim.Adam(other_param_groups, lr=lr)
            else:
                opt_base = torch.optim.SGD(other_param_groups, lr=lr)
            self.optimizer = HybridOptimizer([opt_spec, opt_base])
        else:
            param_groups = [{'params': base_params},
                            {'params': gate_params, 'lr': lr * gate_lr_ratio}]
            if optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(param_groups, lr=lr)
            elif optimizer_type == 'sgd':
                self.optimizer = torch.optim.SGD(param_groups, lr=lr)
            elif optimizer_type == 'adamw':
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd_adamw)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        self.d_x = d_x
        self.d_y = d_y
        self.num_blocks = num_blocks
        self.init_time = None
        self.loss_history = []
        self.y_pred_norm_history = []
        self.y_true_norm_history = []
        self.residual_gate_history = []
        self.sink_score_history = []
        self.sink_rate_history = []
        self.experiment_name = experiment_name
        if load_path is not None:
            self.load_checkpoint(load_path)
        if experiment_name is not None:
            if timing:
                self.init_time = time.time() - start_time
                if print_on:
                    print(f"{experiment_name} initialized in {self.init_time:.2f} seconds. Using device: {self.device}")
            else:
                if print_on:
                    print(f"{experiment_name} initialized. Using device: {self.device}")

    def train(self, batch_size=64, seq_len=80, epochs=20, steps_per_epoch=1,
              data_type='linear', d_hidden=None, function_callable=None, lorenz_kwargs=None, load_lorenz_from=None,
              num_eff=15, sink_padding=None, scheduled_training=False,
              curriculum=None, scheduler_type=None, lr_scale=0.1, step_size_scheduler=10, eta_min=None,
              print_every=None, timing=False, save_path=None):
        """执行训练循环。

        支持：curriculum learning（按 progress 在前 duration_ratio 比例的训练步内逐步放大
        d_x / seq_len）、scheduled training（current_blocks 从 num_eff 渐增至 num_blocks）、
        cosine/step 学习率调度（HybridOptimizer 时分别给两个子 optimizer 建调度器）、
        每 epoch 末尾切 eval 做一次前向以捕获 sink 指标、训练结束保存 checkpoint。

        Args:
            batch_size (int): 批次大小。
            seq_len (int): 序列长度。
            epochs (int): 训练轮数。
            steps_per_epoch (int): 每轮训练步数。
            data_type (str): 数据类型 'linear'/'nonlinear'/'lorenz'。
            d_hidden (int or None): nonlinear 隐藏层维度。
            function_callable (callable or None): nonlinear 非线性函数。
            lorenz_kwargs (dict or None): lorenz 参数（dt/burn_in）。
            load_lorenz_from (str or None): lorenz 离线池路径。
            num_eff (int): 有效层数 T，实际取 min(num_eff, num_blocks)。
            sink_padding (int or None): sink token 组数。
            scheduled_training (bool): 是否渐进增加 current_blocks。
            curriculum (dict or None): 课程学习配置（d_x/seq_len/duration_ratio）。
            scheduler_type (str or None): 'cosine'/'step'/None。
            lr_scale (float): 调度器缩放因子（cosine 的默认 eta_min 比例 / StepLR 的 gamma）。
            step_size_scheduler (int): StepLR 步长。
            eta_min (float or None): cosine 的最小学习率，None 时用 lr_scale 推算。
            print_every (int or None): 每 N 个 epoch 打印一次。
            timing (bool): 是否打印训练耗时。
            save_path (str or None): 训练完成后 checkpoint 保存路径。
        """
        # x_data: [batch_size, k, d_x]
        # y_data: [batch_size, k, d_y]
        self.model.train()
        self.max_torch_vram = 0
        self.max_metal_vram = 0
        self.train_time = None
        dataloader_static_kwargs = dict(batch_size=batch_size, d_x=self.d_x, d_y=self.d_y, device=self.device, data_type=data_type, sink_padding=sink_padding, generator=self.generator, d_hidden=d_hidden, function_callable=function_callable, ood_kwargs=None, lorenz_kwargs=lorenz_kwargs, load_lorenz_from=load_lorenz_from)
        active_seq_len = seq_len
        active_d_x = self.d_x
        active_data_loader = dataloader(seq_len=active_seq_len, valid_d_x=active_d_x, **dataloader_static_kwargs)
        is_hybrid = isinstance(self.optimizer, HybridOptimizer)
        if scheduler_type == 'cosine':
            if is_hybrid:
                eta_min_spec = eta_min if eta_min is not None else lr_scale * self.optimizer.optimizers[0].param_groups[0]['lr']
                scheduler_spec = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer.optimizers[0], T_max=epochs, eta_min=eta_min_spec)
                eta_min_base = eta_min if eta_min is not None else lr_scale * self.optimizer.optimizers[1].param_groups[0]['lr']
                scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer.optimizers[1], T_max=epochs, eta_min=eta_min_base)
                scheduler = [scheduler_spec, scheduler_base]
            else:
                if eta_min is None:
                    eta_min = lr_scale * self.optimizer.param_groups[0]['lr']
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=eta_min)
        elif scheduler_type == 'step':
            if is_hybrid:
                scheduler_spec = torch.optim.lr_scheduler.StepLR(self.optimizer.optimizers[0], step_size=step_size_scheduler, gamma=lr_scale)
                scheduler_base = torch.optim.lr_scheduler.StepLR(self.optimizer.optimizers[1], step_size=step_size_scheduler, gamma=lr_scale)
                scheduler = [scheduler_spec, scheduler_base]
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size_scheduler, gamma=lr_scale)
        num_eff = min(num_eff, self.num_blocks)  # 确保 num_eff 不超过总层数
        if curriculum is None:
            curriculum = {}
        total_steps = epochs * steps_per_epoch
        if timing:
            start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_y_pred_norm = 0.0
            epoch_y_true_norm = 0.0
            if scheduled_training:
                current_blocks = min(num_eff + epoch, self.num_blocks)  # 随着训练的进行，逐渐增加参与误差计算的有效层数
            else:
                current_blocks = self.num_blocks
            for step in range(steps_per_epoch):
                self.optimizer.zero_grad()
                global_step = epoch * steps_per_epoch + step
                if curriculum:
                    progress = min(1.0, global_step / (total_steps * curriculum.get('duration_ratio', 0.5)))
                    if progress == 1.0:
                        curriculum = {}  # 课程结束，重置为普通训练
                        if print_every is not None:
                            print(f"Curriculum ended at epoch {epoch+1}, step {step+1}.")
                    else:
                        d_x_start = curriculum.get('d_x', self.d_x)
                        seq_len_start = curriculum.get('seq_len', seq_len)
                        current_d_x = int(d_x_start + progress * (self.d_x - d_x_start))
                        current_seq_len = int(seq_len_start + progress * (seq_len - seq_len_start))
                else:
                    current_d_x = self.d_x
                    current_seq_len = seq_len
                if current_seq_len != active_seq_len or current_d_x != active_d_x:
                    active_seq_len = current_seq_len
                    active_d_x = current_d_x
                    active_data_loader = dataloader(seq_len=active_seq_len, valid_d_x=active_d_x, **dataloader_static_kwargs)
                x_data, y_data = next(active_data_loader)  # 从数据加载器中获取一个批次的数据
                loss, y_pred_norm, y_true_norm = self.model(x_data, y_data, num_eff=num_eff, current_blocks=current_blocks, is_eval=False, sink_padding=sink_padding)  # 前向传播计算损失 [batch_size, num_eff, seq_len/2, d_y]
                loss = loss.mean()  # 对所有维度求平均得到一个标量损失值
                loss.backward()  # 反向传播计算梯度
                if torch.backends.mps.is_available():
                    self.max_metal_vram = max(self.max_metal_vram, torch.mps.driver_allocated_memory() / (1024 ** 3))
                    self.max_torch_vram = max(self.max_torch_vram, torch.mps.current_allocated_memory() / (1024 ** 3))
                self.optimizer.step()  # 更新模型参数
                epoch_loss += loss.item()
                epoch_y_pred_norm += y_pred_norm
                epoch_y_true_norm += y_true_norm

            self.model.eval()  # 切换到评估模式，记录注意力分数(这样不会进行FlashAttention跳过attention矩阵生成)
            with torch.no_grad():
                _ = self.model(x_data, y_data, num_eff=num_eff, current_blocks=current_blocks, is_eval=False, sink_padding=sink_padding)  # 再次前向传播以获取当前的残差门控参数和注意力分数
                scores = self.model.toy_model.captured_sink_scores
                rates = self.model.toy_model.captured_sink_rates
                if scores and rates:
                    valid_scores = [score for score in scores if score is not None]
                    valid_rates = [rate for rate in rates if rate is not None]
                    avg_sink_score = np.mean(valid_scores) if valid_scores else 0.0
                    avg_sink_rate = np.mean(valid_rates) if valid_rates else 0.0
                else:
                    avg_sink_score, avg_sink_rate = 0.0, 0.0
            self.model.train()  # 切换回训练模式

            if scheduler_type is not None:
                if isinstance(scheduler, list):
                    for s in scheduler:
                        s.step()
                else:
                    scheduler.step()  # 按epoch级别更新学习率
            avg_loss = epoch_loss / steps_per_epoch
            avg_y_pred_norm = epoch_y_pred_norm / steps_per_epoch
            avg_y_true_norm = epoch_y_true_norm / steps_per_epoch
            avg_y_cos = (1 + (avg_y_pred_norm / avg_y_true_norm) ** 2 - (avg_loss / (avg_y_true_norm ** 2))) / (2 * (avg_y_pred_norm / avg_y_true_norm)) if avg_y_true_norm != 0 and avg_y_pred_norm != 0 else 0.0
            avg_y_cos_1 = 1 - avg_y_cos
            self.y_pred_norm_history.append(avg_y_pred_norm)
            self.y_true_norm_history.append(avg_y_true_norm)
            self.loss_history.append(avg_loss)  # 记录损失值
            self.sink_score_history.append(avg_sink_score)
            self.sink_rate_history.append(avg_sink_rate)
            self.residual_gate_history.append(self.model.toy_model.residual_gate.detach().cpu().numpy() if hasattr(self.model.toy_model, 'residual_gate') else None)  # 记录门控参数值
            if print_every is not None and (epoch + 1) % print_every == 0:
                avg_y_cos = (1 + (avg_y_pred_norm / avg_y_true_norm) ** 2 - (avg_loss / (avg_y_true_norm ** 2))) / (2 * (avg_y_pred_norm / avg_y_true_norm)) if avg_y_true_norm != 0 and avg_y_pred_norm != 0 else 0.0
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Norm Ratio: {avg_y_pred_norm/avg_y_true_norm if avg_y_true_norm != 0 else 0.0:.4f}, Norm Cosine Similarity: {avg_y_cos:.4f}, Avg Sink Score: {avg_sink_score:.4f}, Avg Sink Rate: {avg_sink_rate:.4f}")
        if timing:
            self.train_time = time.time() - start_time
            if print_every is not None:
                print(f"Training completed in {self.train_time:.2f} seconds.")
        self.residual_gate_values = self.residual_gate_history[-1]
        self.final_loss = self.loss_history[-1]
        if save_path is not None:
            self.save_checkpoint(save_path)

    def evaluate(self, eval_name='eval', batch_size=64, seq_len=80, data_type='linear', loss_type='mse', sink_padding=1, d_hidden=None, function_callable=None, ood_kwargs=None, lorenz_kwargs=None):
        """评估模型性能。

        生成 seq_len+2 的数据（多 2 以预测最后位置），取最后一层最后一位置的预测值与
        真实值计算 loss / norm / cosine 等指标，并捕获 sink 指标与注意力矩阵。所有结果
        以 ``{eval_name}_*`` 为 key 存入 ``self.eval_results``。

        Args:
            eval_name (str): 评估名称，用作结果 key 前缀。
            batch_size (int): 批次大小。
            seq_len (int): 评估序列长度（内部会 +2）。
            data_type (str): 数据类型。
            loss_type (str): 'mse' 或 'l1'。
            sink_padding (int): sink padding 组数。
            d_hidden (int or None): nonlinear 隐藏层维度。
            function_callable (callable or None): nonlinear 非线性函数。
            ood_kwargs (dict or None): OOD 配置（可含 seq_len_scale 做长度外推）。
            lorenz_kwargs (dict or None): lorenz 参数（评估时实时生成，不用离线池）。
        """
        self.model.eval()
        if ood_kwargs is None:
            ood_kwargs = {}
        if 'seq_len_scale' in ood_kwargs:
            seq_len = int(seq_len * ood_kwargs['seq_len_scale'])  # 调整序列长度，制造OOD数据
        with torch.no_grad():
            # 在评估模式下，直接使用所有层的输出进行预测(因为已经no_grad了，所以不需要防止梯度爆炸了)
            # 生成一个评估用的数据批次，注意这里的seq_len要比训练时大2，因为我们需要预测最后一个位置的y值
            x_eval, y_eval = next(dataloader(batch_size=batch_size, seq_len=seq_len+2, d_x=self.d_x, d_y=self.d_y, device=self.device, data_type=data_type, sink_padding=sink_padding,
                                             generator=self.generator, d_hidden=d_hidden, function_callable=function_callable, ood_kwargs=ood_kwargs, lorenz_kwargs=lorenz_kwargs, load_lorenz_from=None))  # Lorenz在评估时不使用离线数据池，而是直接生成新的数据
            y_pred = self.model(x_eval, y_eval, num_eff=self.num_blocks, current_blocks=self.num_blocks, is_eval=True, sink_padding=sink_padding)  # [batch_size, d_y]
            # 计算 loss
            if loss_type == 'mse':
                loss = F.mse_loss(y_pred, y_eval[:, -1, :]).item()
            elif loss_type == 'l1':
                loss = F.l1_loss(y_pred, y_eval[:, -1, :]).item()
            # 计算预测值和真实值的范数以及它们的比值
            y_pred_norm = torch.sqrt(torch.mean(y_pred ** 2)).item()
            y_true_norm = torch.sqrt(torch.mean(y_eval[:, -1, :] ** 2)).item()
            y_norm_ratio = y_pred_norm / y_true_norm if y_true_norm != 0 else 0.0
            y_norm_ratio_abs = abs(y_norm_ratio - 1.0)
            y_cos = (1 + y_norm_ratio**2 - loss/(y_true_norm**2)) / (2 * y_norm_ratio) if y_true_norm != 0 and y_norm_ratio != 0 else 0.0
            y_cos_1 = 1 - y_cos
            print(f"[{eval_name}] [{self.experiment_name}]Loss: {loss:.4f}, Norm Ratio: {y_norm_ratio:.4f}, Norm Cosine Similarity: {y_cos:.4f}")
            self.eval_results.update({
                f'{eval_name}_loss': loss,
                f'{eval_name}_y_pred_norm': y_pred_norm,
                f'{eval_name}_y_true_norm': y_true_norm,
                f'{eval_name}_y_norm_ratio': y_norm_ratio,
                f'{eval_name}_y_norm_ratio_abs': y_norm_ratio_abs,
                f'{eval_name}_y_cos': y_cos,
                f'{eval_name}_y_cos_1': y_cos_1,
                f'{eval_name}_sink_scores': self.model.toy_model.captured_sink_scores,  # list[float]: 长度为num_blocks，每个元素是对应的一层中，所有注意力头分配给第0个位置的平均注意力权重
                f'{eval_name}_sink_rates': self.model.toy_model.captured_sink_rates,  # list[float]: 长度为num_blocks，每个元素是对应的一层中，分配给第0个位置的注意力权重比例>sink_threshold的注意力头的比例
                f'{eval_name}_captured_attention': self.model.toy_model.captured_attention
            })

    def get_results(self, result_list: list[str]):
        """提取指定指标的实验数据。

        根据 gate 是标量（1 维）还是向量（2 维）自动派生额外的统计量（均值、标准差、
        相对变化等）。请求但不可用的 key 返回 None（不报错）。

        Args:
            result_list (list[str]): 需要提取的指标名列表。

        Returns:
            dict: {指标名: 数据}，请求但缺失的 key 对应值为 None。
        """
        all_results = {
            'init_time': getattr(self, 'init_time', None),                                       # float: 实验初始化时间（秒）
            'train_time': getattr(self, 'train_time', None),                                     # float: 训练时间（秒)
            'loss_history': getattr(self, 'loss_history', None),                                 # list[float]: 训练过程中每个epoch的损失值
            'y_pred_norm_history': getattr(self, 'y_pred_norm_history', None),                   # list[float]: 训练过程中每个epoch的预测值范数
            'y_true_norm_history': getattr(self, 'y_true_norm_history', None),                   # list[float]: 训练过程中每个epoch的真实值范数
            'y_norm_error_history': [pred - true for pred, true in zip(getattr(self, 'y_pred_norm_history', []), getattr(self, 'y_true_norm_history', []))] if getattr(self, 'y_pred_norm_history', None) is not None and getattr(self, 'y_true_norm_history', None) is not None else None,                        # list[float]: 训练过程中每个epoch的预测值范数与真实值范数之差的绝对值
            'y_norm_ratio_history': [pred / true if true != 0 else None for pred, true in zip(getattr(self, 'y_pred_norm_history', []), getattr(self, 'y_true_norm_history', []))] if getattr(self, 'y_pred_norm_history', None) is not None and getattr(self, 'y_true_norm_history', None) is not None else None,  # list[float]: 训练过程中每个epoch的预测值范数与真实值范数之比
            'y_norm_ratio_abs_history': [abs(pred / true - 1.0) if true != 0 else None for pred, true in zip(getattr(self, 'y_pred_norm_history', []), getattr(self, 'y_true_norm_history', []))] if getattr(self, 'y_pred_norm_history', None) is not None and getattr(self, 'y_true_norm_history', None) is not None else None,  # list[float]: 训练过程中每个epoch的预测值范数与真实值范数之比的绝对值偏差
            'y_cos_history': [(1 + (pred / true)**2 - (loss / (true**2))) / (2 * (pred / true)) if true != 0 and pred != 0 else None for pred, true, loss in zip(getattr(self, 'y_pred_norm_history', []), getattr(self, 'y_true_norm_history', []), getattr(self, 'loss_history', []))] if getattr(self, 'y_pred_norm_history', None) is not None and getattr(self, 'y_true_norm_history', None) is not None and getattr(self, 'loss_history', None) is not None else None,  # list[float]: 训练过程中每个epoch的预测值与真实值的余弦相似度
            'y_cos_1_history': [1 - (1 + (pred / true)**2 - (loss / (true**2))) / (2 * (pred / true)) if true != 0 and pred != 0 else None for pred, true, loss in zip(getattr(self, 'y_pred_norm_history', []), getattr(self, 'y_true_norm_history', []), getattr(self, 'loss_history', []))] if getattr(self, 'y_pred_norm_history', None) is not None and getattr(self, 'y_true_norm_history', None) is not None and getattr(self, 'loss_history', None) is not None else None,  # list[float]: 训练过程中每个epoch的预测值与真实值的余弦相似度的补数
            'sink_score_history': getattr(self, 'sink_score_history', None),                     # list[float]: 长度为epochs，每个元素为每个epoch的平均（对num_blocks,steps_per_epoch取平均）sink score
            'sink_rate_history': getattr(self, 'sink_rate_history', None),                       # list[float]: 长度为epochs，每个元素为每个epoch的平均（对num_blocks,steps_per_epoch取平均）sink rate
            'final_loss': self.loss_history[-1] if getattr(self, 'loss_history', None) else None,  # float: 最终的损失值
            'final_y_pred_norm': self.y_pred_norm_history[-1] if getattr(self, 'y_pred_norm_history', None) else None,               # float: 最终的预测值范数
            'final_y_true_norm': self.y_true_norm_history[-1] if getattr(self, 'y_true_norm_history', None) else None,               # float: 最终的真实值范数
            'residual_gate_history': getattr(self, 'residual_gate_history', None),               # list[np.array[2]或np.array[d_model, 2]]: 训练过程中每个epoch的残差门控参数值
            'final_residual_gate': getattr(self, 'residual_gate_values', None),                  # np.array[2]或np.array[d_model, 2]: 最终的残差门控参数值
            'final_sink_score': self.sink_score_history[-1] if getattr(self, 'sink_score_history', None) else None,                 # float: 最后一个epoch的sink score
            'final_sink_rate': self.sink_rate_history[-1] if getattr(self, 'sink_rate_history', None) else None,                   # float: 最后一个epoch的sink rate
            'captured_sink_scores': getattr(self.model.toy_model, 'captured_sink_scores', None),  # list[float]: 同eval_results中的sink_scores，只不过取自训练过程中最后一次前向传播的捕获值
            'captured_sink_rates': getattr(self.model.toy_model, 'captured_sink_rates', None),    # list[float]: 同eval_results中的sink_rates，只不过取自训练过程中最后一次前向传播的捕获值
            'captured_attention': getattr(self.model.toy_model, 'captured_attention', None)      # list[np.array[batch_size, num_heads, seq_len, seq_len]]: 训练时最后一次前向传播捕获的注意力权重
        }
        if hasattr(self, 'eval_results'):
            all_results.update(self.eval_results)  # 将评估结果添加到 all_results 中
        if any("residual_gate" in key for key in result_list) and self.residual_gate_values is not None:
            if len(self.residual_gate_values.shape) == 1:
                all_results['residual_gate_history_a'] = [gate[0] for gate in self.residual_gate_history]
                all_results['residual_gate_history_b'] = [gate[1] for gate in self.residual_gate_history]
                all_results['residual_gate_history_a_relative'] = [gate[0] - self.residual_gate_history[0][0] for gate in self.residual_gate_history]
                all_results['residual_gate_history_b_relative'] = [gate[1] - self.residual_gate_history[0][1] for gate in self.residual_gate_history]
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
                all_results['residual_gate_history_a_mean_relative'] = [gate[:, 0].mean() - self.residual_gate_history[0][:, 0].mean() for gate in self.residual_gate_history]
                all_results['residual_gate_history_b_mean_relative'] = [gate[:, 1].mean() - self.residual_gate_history[0][:, 1].mean() for gate in self.residual_gate_history]
                all_results['residual_gate_history_a_std'] = [gate[:, 0].std() for gate in self.residual_gate_history]
                all_results['residual_gate_history_b_std'] = [gate[:, 1].std() for gate in self.residual_gate_history]
                all_results['final_residual_gate_a_mean'] = self.residual_gate_values[:, 0].mean()
                all_results['final_residual_gate_b_mean'] = self.residual_gate_values[:, 1].mean()
                all_results['final_residual_gate_a_std'] = self.residual_gate_values[:, 0].std()
                all_results['final_residual_gate_b_std'] = self.residual_gate_values[:, 1].std()
        if result_list is not None:
            return {key: all_results.get(key) for key in result_list}

    def offload_to_cpu(self):
        """将模型与优化器状态转移到 CPU 以节省 GPU 显存，并置 is_offloaded=True。"""
        self.model.to('cpu')
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        self.is_offloaded = True

    def load_to_device(self):
        """将模型与优化器状态从 CPU 恢复到原始设备，与 offload_to_cpu 配对。"""
        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.is_offloaded = False

    def save_checkpoint(self, path='auto'):
        """保存实验的完整状态（模型、优化器、各历史指标）。

        Args:
            path (str): 保存路径；'auto' 时取 ``saved_checkpoints/<safe_experiment_name>.pth``。
        """
        if path == 'auto':
            exp_name = self.experiment_name if self.experiment_name is not None else "Experiment"
            safe_name = exp_name.replace(' ', '_').replace('(', '').replace(')', '')
            path = f'saved_checkpoints/{safe_name}.pth'
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'loss_history': getattr(self, 'loss_history', []),
            'y_pred_norm_history': getattr(self, 'y_pred_norm_history', []),
            'y_true_norm_history': getattr(self, 'y_true_norm_history', []),
            'sink_score_history': getattr(self, 'sink_score_history', []),
            'sink_rate_history': getattr(self, 'sink_rate_history', []),
            'residual_gate_history': getattr(self, 'residual_gate_history', []),
        }
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint 已安全保存至: {path}")

    def load_checkpoint(self, path='auto'):
        """加载实验的完整状态。

        支持两种格式：含 ``model_state_dict`` 的完整 checkpoint（恢复模型+历史+优化器），
        或纯权重 state_dict（兼容历史版本，历史记录清空）。

        Args:
            path (str): 加载路径；'auto' 时取 ``saved_checkpoints/<safe_experiment_name>.pth``。
        """
        if path == 'auto':
            exp_name = self.experiment_name if self.experiment_name is not None else "Experiment"
            safe_name = exp_name.replace(' ', '_').replace('(', '').replace(')', '')
            path = f'saved_checkpoints/{safe_name}.pth'
        if not os.path.exists(path):
            print(f"⚠️ 找不到文件 {path}，跳过加载。")
            return
        # 统一先加载到 CPU，后续再通过 load_to_device() 转移
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 恢复历史记录（方便画图断点续联）
            self.loss_history = checkpoint.get('loss_history', [])
            self.y_pred_norm_history = checkpoint.get('y_pred_norm_history', [])
            self.y_true_norm_history = checkpoint.get('y_true_norm_history', [])
            self.sink_score_history = checkpoint.get('sink_score_history', [])
            self.sink_rate_history = checkpoint.get('sink_rate_history', [])
            self.residual_gate_history = checkpoint.get('residual_gate_history', [])
            # 恢复优化器状态
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("检测到纯权重 state_dict 文件（或历史版本），直接加载 model_state_dict...")
            self.model.load_state_dict(checkpoint)
            self.loss_history = []
            self.y_pred_norm_history = []
            self.y_true_norm_history = []
            self.sink_score_history = []
            self.sink_rate_history = []
            self.residual_gate_history = []
        print(f"🔄 成功从 {path} 恢复实验状态。")
