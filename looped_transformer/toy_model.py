"""ToyModel：Looped Transformer 核心引擎。

实现权重共享的循环 Transformer，支持残差门控 (a·x + b·x_0)、截断反向传播 (num_eff)、
初始状态选择 (x_init)、以及 AttentionProbe / SinkMetricsProbe 的无感集成。
"""

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from .core.position_encoding import APE, LearnedAPE
from .core.probes import AttentionProbe, SinkMetricsProbe


class ToyModel(nn.Module):
    """Looped Transformer 引擎。

    支持两种模式：
    - ``loop=True``：所有迭代共享同一个 TransformerBlock 的权重（权重共享）。
    - ``loop=False``：每层独立的 TransformerBlock（标准深层 Transformer）。

    核心公式：``x_l = TransformerBlock(a · x_{l-1} + b · x_0)``，其中 (a, b) 为残差
    门控参数，x_0 为加入位置编码后的初始输入。

    通过 ``num_eff`` 控制截断反向传播：只有最后 num_eff 层参与损失计算与梯度回传，
    前 ``current_blocks - num_eff`` 层在 ``b_0`` 处被 detach 释放，防止梯度爆炸。

    初始状态由 ``x_init`` 决定：'prompt' 直接用 x_0；'zero' 用全零向量（测试模型能否
    从零开始学习）。会自动在注意力上注册 AttentionProbe 与 SinkMetricsProbe。

    Attributes:
        transformer_block (TransformerBlock or nn.ModuleList): 循环块或块列表。
        residual_gate (torch.Tensor or nn.Parameter): 残差门控 (a, b)，可为固定 buffer、
            可学习标量或可学习向量。
        ape (nn.ModuleList): 绝对位置编码模块列表（APE/LearnedAPE）。
        probe (AttentionProbe or list): 注意力探针。
        sink_metrics_probe (SinkMetricsProbe or list): sink 指标探针。
        captured_attention (list or None): 最近一次前向传播捕获的注意力矩阵。
        captured_sink_scores (list or None): 最近一次前向传播的 sink score。
        captured_sink_rates (list or None): 最近一次前向传播的 sink rate。
    """

    def __init__(self, num_blocks, num_heads, d_model, max_seq_len=4096, x_init='prompt',
                 norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape', b_rope_or_upe=10000, head_ratio_upe=2, sink_threshold=0.3,
                 loop=True, residual_gate=(1, 1), residual_gate_type='fixed', residual_random=(1, 0.1)):
        """初始化 ToyModel。

        Args:
            num_blocks (int): 总迭代层数 b。
            num_heads (int): 注意力头数。
            d_model (int): 模型维度。
            max_seq_len (int): 最大序列长度。
            x_init (str): 初始状态来源，'prompt'（用 x_0）或 'zero'（全零）。
            norm_type (str): 归一化类型，'layernorm' 或 'rmsnorm'。
            ffn_type (str): FFN 类型，'gelu' 或 'swiglu'。
            pe_type (str or list[str]): 位置编码类型。
            b_rope_or_upe (int): RoPE/MS_UPE 基频。
            head_ratio_upe (int): MS_UPE 头倍率。
            sink_threshold (float): SinkMetricsProbe 判断 sink 的阈值。
            loop (bool): 是否权重共享。True 表示所有层共享同一组权重。
            residual_gate (tuple or str): 残差门控初始值，tuple 如 (a, b)，
                或 'random' 表示从 residual_random 指定的高斯分布采样。
            residual_gate_type (str): 门控类型，'fixed'、'learnable_scalar'、
                'learnable_vector'。
            residual_random (tuple): (mean, std)，当 residual_gate='random' 时的高斯分布参数。

        Raises:
            ValueError: residual_gate 为 (0,0) 时（输入被完全阻断，模型无法学习）。
        """
        super().__init__()
        self.loop = loop
        self.x_init = x_init
        if residual_gate == (0, 0):
            raise ValueError("residual_gate cannot be (0,0) because then the model would be unable to learn anything (the input would be completely blocked)")
        if residual_gate[1] == 0 and x_init == 'zero':
            self.x_init = 'prompt'
            print("Warning: x_init is set to 'prompt' because residual_gate[1] is 0, which would make the model completely blind.")

        if loop:
            self.transformer_block = TransformerBlock(num_heads=num_heads, d_model=d_model, max_seq_len=max_seq_len, norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type, b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe)  # 这里保证了各层权重始终相同
            self.probe = AttentionProbe()
            self.sink_metrics_probe = SinkMetricsProbe(threshold=sink_threshold)
            self.transformer_block.attention.register_forward_hook(self.probe)
            self.transformer_block.attention.register_forward_hook(self.sink_metrics_probe)
            self.captured_attention = None
            self.captured_sink_scores = None
            self.captured_sink_rates = None
        else:
            self.transformer_block = nn.ModuleList([
                TransformerBlock(num_heads=num_heads, d_model=d_model, max_seq_len=max_seq_len, norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type, b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe)
                for _ in range(num_blocks)
            ])
            self.probe = [AttentionProbe() for _ in range(num_blocks)]
            self.sink_metrics_probe = [SinkMetricsProbe(threshold=sink_threshold) for _ in range(num_blocks)]
            for block, probe, sink_probe in zip(self.transformer_block, self.probe, self.sink_metrics_probe):
                block.attention.register_forward_hook(probe)
                block.attention.register_forward_hook(sink_probe)

        residual_random = torch.tensor(residual_random, dtype=torch.float32)
        if residual_gate == 'random':
            residual_gate_tensor = torch.randn(2) * residual_random[1] + residual_random[0]  # 随机初始化门控参数，初始正态分布
        else:
            residual_gate_tensor = torch.tensor(residual_gate, dtype=torch.float32)  # 将输入的元组转换为张量，方便后续计算
        if residual_gate_type == 'fixed':
            self.register_buffer('residual_gate', residual_gate_tensor)  # 固定的门控参数，注册为buffer使其成为模型状态的一部分，但不参与梯度更新
        elif residual_gate_type == 'learnable_scalar':
            self.residual_gate = nn.Parameter(torch.ones(2) * residual_gate_tensor)  # 可学习的标量门控参数，初始值为用户指定的值
        elif residual_gate_type == 'learnable_vector':
            if residual_gate == 'random':
                self.residual_gate = nn.Parameter(torch.randn(d_model, 2) * residual_random[1] + residual_random[0])  # 可学习的向量门控参数，初始值为随机向量
            else:
                self.residual_gate = nn.Parameter(torch.ones(d_model, 2) * residual_gate_tensor)  # 可学习的向量门控参数，初始值为全1向量
        if loop:
            pe_type = self.transformer_block.attention.pe_type
        else:
            pe_type = self.transformer_block[0].attention.pe_type
        self.ape = nn.ModuleList()
        for pe in pe_type:
            if pe == 'ape':
                self.ape.append(APE(d_model, max_seq_len=max_seq_len))
            elif pe == 'learned_ape':
                self.ape.append(LearnedAPE(d_model, max_seq_len=max_seq_len))
            elif pe in ['rope', 'ms_upe']:
                pass  # RoPE和MS_UPE在MultiHeadAttention内部处理位置编码，不需要单独的模块在这里处理

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type

    def forward(self, x_0, num_eff=15, current_blocks=None):
        """前向传播：执行 current_blocks 次循环迭代。

        前 ``current_blocks - num_eff`` 层的计算图会在 ``b_0`` 处被 detach 释放，
        只有最后 num_eff 层的输出被收集并参与梯度计算。

        Args:
            x_0 (torch.Tensor): 初始输入 [batch_size, seq_len, d_model]。
                APE/LearnedAPE 会先加到 x_0 上；若 x_init='zero' 则初始状态为全零。
            num_eff (int): 有效层数 T，即参与梯度回传的层数。
            current_blocks (int or None): 实际执行的迭代数。None 表示使用 num_blocks
                （用于 scheduled training 时逐步增大）。

        Returns:
            torch.Tensor: 最后 num_eff 层的输出堆叠 [batch_size, num_eff, seq_len, d_model]。
        """
        # num_eff 是需要加入误差计算的有效层数，也即理论公式中的 T
        # x 的形状为 [batch_size, seq_len, d_model]
        a = self.residual_gate[..., 0]
        b = self.residual_gate[..., 1]
        for ape_module in self.ape:
            x_0 = ape_module(x_0)  # 绝对位置编码只加在最开始输入处
        if self.x_init == 'prompt':
            x = x_0
        elif self.x_init == 'zero':
            x = torch.zeros_like(x_0)  # 直接用全零向量作为初始输入，测试模型是否能从零开始学习
        else:
            raise ValueError(f"Invalid x_init value: {self.x_init}. Expected 'prompt' or 'zero'.")
        if current_blocks is None:
            current_blocks = self.num_blocks
        b_0 = max(0, current_blocks - num_eff)  # 计算需要加入误差计算的有效层数对应的起始层索引
        outputs = []
        if self.loop:
            self.probe.reset()  # 每次前向传播前重置捕获的数据，避免混淆不同次前向传播的注意力矩阵
            self.sink_metrics_probe.reset()  # 同样重置 sink metrics 的捕获数据
            for i in range(current_blocks):
                if i == b_0:
                    x = x.detach()  # 释放前面的计算图
                    x.requires_grad_(True)
                x = self.transformer_block(a * x + b * x_0)
                if i >= b_0:
                    outputs.append(x)
            self.captured_attention = list(self.probe.captured_data)  # 显式copy一份数据，避免后续被修改
            self.captured_sink_scores = list(self.sink_metrics_probe.captured_scores)
            self.captured_sink_rates = list(self.sink_metrics_probe.captured_rates)
        else:
            for probe in self.probe:
                probe.reset()
            for sink_probe in self.sink_metrics_probe:
                sink_probe.reset()
            for i, block in enumerate(self.transformer_block[:current_blocks]):  # 只迭代当前有效层数，避免不必要的计算
                if i == b_0:
                    x = x.detach()  # 释放前面的计算图
                    x.requires_grad_(True)
                x = block(a * x + b * x_0)  # x: [batch_size, seq_len, d_model]
                if i >= b_0:
                    outputs.append(x)
            self.captured_attention = [probe.captured_data[-1] if probe.captured_data else None for probe in self.probe]
            self.captured_sink_scores = [sink_probe.captured_scores[-1] if sink_probe.captured_scores else None for sink_probe in self.sink_metrics_probe]
            self.captured_sink_rates = [sink_probe.captured_rates[-1] if sink_probe.captured_rates else None for sink_probe in self.sink_metrics_probe]
        if self.loop:
            if hasattr(self.transformer_block.attention, 'captured_attention'):
                self.transformer_block.attention.captured_attention = None
        else:
            for block in self.transformer_block:
                if hasattr(block.attention, 'captured_attention'):
                    block.attention.captured_attention = None
        outputs = torch.stack(outputs, dim=1)  # [batch_size, num_eff, seq_len, d_model]
        return outputs
