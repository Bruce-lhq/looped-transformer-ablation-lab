"""Transformer Block 模块。

标准 Pre-Norm TransformerBlock：归一化 → MultiHeadAttention → 残差 →
归一化 → FFN (GELU 或 SwiGLU) → 残差。SwiGLU 本身见 ``core.swiglu``。
"""

import torch.nn as nn

from .attention import MultiHeadAttention
from .core.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """标准 Transformer Block（Pre-Norm 架构）。

    数据处理流程：
    1. LayerNorm/RMSNorm → MultiHeadAttention → 残差连接
    2. LayerNorm/RMSNorm → FFN (GELU/SwiGLU) → 残差连接

    Attributes:
        norm1, norm2 (nn.Module): 两次归一化层。
        attention (MultiHeadAttention): 多头注意力。
        ffn (nn.Module): 前馈网络（GELU 序列或 SwiGLU）。
    """

    def __init__(self, num_heads, d_model, max_seq_len=4096, multiplier=4, norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape', b_rope_or_upe=10000, head_ratio_upe=2):
        """初始化 TransformerBlock。

        Args:
            num_heads (int): 注意力头数。
            d_model (int): 模型维度。
            max_seq_len (int): 最大序列长度。
            multiplier (int): FFN 隐藏层倍数（d_hidden = d_model * multiplier）。
            norm_type (str): 归一化类型，'layernorm' 或 'rmsnorm'。
            ffn_type (str): 前馈网络类型，'gelu' 或 'swiglu'。
            pe_type (str or list[str]): 位置编码类型，透传给 MultiHeadAttention。
            b_rope_or_upe (int): RoPE/MS_UPE 基频参数。
            head_ratio_upe (int): MS_UPE 头倍率。
        """
        super().__init__()
        if norm_type == 'rmsnorm':
            self.norm1 = nn.RMSNorm(d_model)  # 第一次归一化
            self.norm2 = nn.RMSNorm(d_model)  # 第二次归一化
        elif norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)  # 第一次归一化
            self.norm2 = nn.LayerNorm(d_model)  # 第二次归一化
        self.attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model, max_seq_len=max_seq_len, pe_type=pe_type, b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe)
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model=d_model, d_hidden=d_model * multiplier)
        elif ffn_type == 'gelu':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * multiplier),
                nn.GELU(),
                nn.Linear(d_model * multiplier, d_model)
            )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: 输出 [batch_size, seq_len, d_model]。
        """
        # 第一次归一化 Pre-Norm
        x_norm1 = self.norm1(x)
        # 全局信息交互 Multi-Head Attention (MHA)
        h = self.attention(x_norm1)
        # 第一次残差连接
        x1 = x + h
        # 第二次归一化
        x_norm2 = self.norm2(x1)
        # 前馈网络 Feed-Forward Network (FFN)
        x_out = self.ffn(x_norm2)
        # 第二次残差连接
        x2 = x1 + x_out
        return x2
