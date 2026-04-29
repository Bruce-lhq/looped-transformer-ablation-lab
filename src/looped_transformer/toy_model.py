"""ToyModel：Looped Transformer 核心引擎。

实现权重共享的循环 Transformer，支持残差门控 (a*x + b*x_0) 和截断反向传播 (num_eff)。
包含 AttentionProbe 用于捕获各层的注意力权重矩阵。
"""

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from .position_encoding import APE, LearnedAPE


class AttentionProbe:
    """注意力探针。

    作为 PyTorch forward hook 注册到 MultiHeadAttention 上，在每次前向传播后
    无感截取模块的 captured_attention 属性并存储为 numpy 数组。

    Attributes:
        captured_data (list): 收集到的注意力矩阵列表，每个元素为 numpy 数组或 None。
    """

    def __init__(self):
        """初始化 AttentionProbe。"""
        self.captured_data = []

    def __call__(self, module, input, output):
        """Hook 回调函数。

        Args:
            module (nn.Module): 被 hook 的模块（MultiHeadAttention 实例）。
            input: 模块输入（未使用）。
            output: 模块输出（未使用）。
        """
        if getattr(module, 'captured_attention', None) is not None:
            attention = module.captured_attention.detach().cpu().numpy()
            self.captured_data.append(attention)
        else:
            self.captured_data.append(None)

    def reset(self):
        """清空已捕获的数据，在每次前向传播前调用以避免混淆。"""
        self.captured_data = []


class ToyModel(nn.Module):
    """Looped Transformer 引擎。

    支持两种模式：
    - loop=True：所有迭代共享同一个 TransformerBlock 的权重（权重共享）。
    - loop=False：每层独立的 TransformerBlock（标准深层 Transformer）。

    核心公式：x_{l} = TransformerBlock(a * x_{l-1} + b * x_0)
    其中 (a, b) 为残差门控参数，x_0 为加入位置编码后的初始输入。

    通过 num_eff 参数控制截断反向传播：只有最后 num_eff 层参与损失计算和梯度回传，
    前 b - num_eff 层的计算图被 detach 释放，有效防止梯度爆炸。

    Attributes:
        transformer_block (TransformerBlock or nn.ModuleList): 循环块或块列表。
        residual_gate (torch.Tensor or nn.Parameter): 残差门控 (a, b)，
            可为固定 buffer、可学习标量或可学习向量。
        ape (nn.ModuleList): 绝对位置编码模块列表。
        probe (AttentionProbe or list): 注意力探针。
        captured_attention (list or None): 最近一次前向传播捕获的注意力矩阵。
    """

    def __init__(self, num_blocks, num_heads, d_model, max_seq_len=4096,
                 norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape',
                 b_rope_or_upe=10000, head_ratio_upe=2,
                 loop=True, residual_gate=(1, 1), residual_gate_type='fixed',
                 residual_random=(1, 0.1)):
        """初始化 ToyModel。

        Args:
            num_blocks (int): 总迭代层数 b。
            num_heads (int): 注意力头数。
            d_model (int): 模型维度。
            max_seq_len (int): 最大序列长度。
            norm_type (str): 归一化类型，'layernorm' 或 'rmsnorm'。
            ffn_type (str): FFN 类型，'gelu' 或 'swiglu'。
            pe_type (str or list[str]): 位置编码类型。
            b_rope_or_upe (int): RoPE/MS_UPE 基频。
            head_ratio_upe (int): MS_UPE 头倍率。
            loop (bool): 是否权重共享。True 表示所有层共享同一组权重。
            residual_gate (tuple or str): 残差门控初始值。tuple 如 (a, b)，
                或 'random' 表示从 residual_random 指定的高斯分布采样。
            residual_gate_type (str): 门控类型，'fixed'（不可学习）、
                'learnable_scalar'（可学习标量）、'learnable_vector'（可学习逐维向量）。
            residual_random (tuple): (mean, std)，当 residual_gate='random' 时
                的高斯分布参数。
        """
        super().__init__()
        self.loop = loop
        if loop:
            self.transformer_block = TransformerBlock(
                num_heads, d_model, max_seq_len=max_seq_len,
                norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type,
                b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe,
            )
            self.probe = AttentionProbe()
            self.transformer_block.attention.register_forward_hook(self.probe)
            self.captured_attention = None
        else:
            self.transformer_block = nn.ModuleList([
                TransformerBlock(
                    num_heads, d_model, max_seq_len=max_seq_len,
                    norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type,
                    b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe,
                )
                for _ in range(num_blocks)
            ])
            self.probe = [AttentionProbe() for _ in range(num_blocks)]
            for block, probe in zip(self.transformer_block, self.probe):
                block.attention.register_forward_hook(probe)

        residual_random = torch.tensor(residual_random, dtype=torch.float32)
        if residual_gate == 'random':
            residual_gate_tensor = torch.randn(2) * residual_random[1] + residual_random[0]
        else:
            residual_gate_tensor = torch.tensor(residual_gate, dtype=torch.float32)

        if residual_gate_type == 'fixed':
            self.register_buffer('residual_gate', residual_gate_tensor)
        elif residual_gate_type == 'learnable_scalar':
            self.residual_gate = nn.Parameter(torch.ones(2) * residual_gate_tensor)
        elif residual_gate_type == 'learnable_vector':
            if residual_gate == 'random':
                self.residual_gate = nn.Parameter(
                    torch.randn(d_model, 2) * residual_random[1] + residual_random[0]
                )
            else:
                self.residual_gate = nn.Parameter(torch.ones(d_model, 2) * residual_gate_tensor)

        pe_type_list = self.transformer_block.attention.pe_type if loop else self.transformer_block[0].attention.pe_type
        self.ape = nn.ModuleList()
        for pe in pe_type_list:
            if pe == 'ape':
                self.ape.append(APE(d_model, max_seq_len=max_seq_len))
            elif pe == 'learned_ape':
                self.ape.append(LearnedAPE(d_model, max_seq_len=max_seq_len))

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type_list

    def forward(self, x_0, num_eff=15, current_blocks=None):
        """前向传播：执行 num_blocks 次循环迭代。

        前 (current_blocks - num_eff) 层的计算图会被 detach 释放，
        只有最后 num_eff 层的输出被收集并参与梯度计算。

        Args:
            x_0 (torch.Tensor): 初始输入 [batch_size, seq_len, d_model]。
            num_eff (int): 有效层数 T，即参与梯度回传的层数。
            current_blocks (int or None): 实际执行的迭代数。None 表示使用 num_blocks。

        Returns:
            torch.Tensor: 最后 num_eff 层的输出堆叠
                [batch_size, num_eff, seq_len, d_model]。
        """
        a = self.residual_gate[..., 0]
        b = self.residual_gate[..., 1]
        for ape_module in self.ape:
            x_0 = ape_module(x_0)
        x = x_0
        if current_blocks is None:
            current_blocks = self.num_blocks
        b_0 = max(0, current_blocks - num_eff)
        outputs = []
        if self.loop:
            self.probe.reset()
            for i in range(current_blocks):
                if i == b_0:
                    x = x.detach()
                    x.requires_grad_(True)
                x = self.transformer_block(a * x + b * x_0)
                if i >= b_0:
                    outputs.append(x)
            self.captured_attention = list(self.probe.captured_data)
        else:
            for probe in self.probe:
                probe.reset()
            for i, block in enumerate(self.transformer_block):
                if i == b_0:
                    x = x.detach()
                    x.requires_grad_(True)
                x = block(a * x + b * x_0)
                if i >= b_0:
                    outputs.append(x)
            self.captured_attention = [probe.captured_data for probe in self.probe]
        outputs = torch.stack(outputs, dim=1)
        return outputs
