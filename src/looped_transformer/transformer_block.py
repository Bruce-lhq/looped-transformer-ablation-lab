"""Transformer Block 模块。

包含 SwiGLU 前馈网络和标准 TransformerBlock（Pre-Norm + MHA + FFN + 双残差连接）。
"""

import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈网络。

    使用 SiLU 激活的门控线性单元，相比 GELU 在部分任务上表现更好。
    计算方式：W2(silu(W(x)) * V(x))。

    Attributes:
        W, V (nn.Linear): 两个独立的升维投影。
        W2 (nn.Linear): 降维投影。
    """

    def __init__(self, d_model, d_hidden):
        """初始化 SwiGLU。

        Args:
            d_model (int): 输入/输出维度。
            d_hidden (int): 隐藏层维度（通常为 d_model * multiplier）。
        """
        super().__init__()
        self.W = nn.Linear(d_model, d_hidden, bias=False)
        self.V = nn.Linear(d_model, d_hidden, bias=False)
        self.W2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: 输出 [batch_size, seq_len, d_model]。
        """
        x1 = F.silu(self.W(x))
        x2 = self.V(x)
        return self.W2(x1 * x2)


class TransformerBlock(nn.Module):
    """标准 Transformer Block（Pre-Norm 架构）。

    数据处理流程：
    1. LayerNorm/RMSNorm → MultiHeadAttention → 残差连接
    2. LayerNorm/RMSNorm → FFN (GELU/SwiGLU) → 残差连接

    Attributes:
        norm1, norm2 (nn.Module): 两次归一化层。
        attention (MultiHeadAttention): 多头注意力。
        ffn (nn.Module): 前馈网络（GELU 或 SwiGLU）。
    """

    def __init__(self, num_heads, d_model, max_seq_len=4096, multiplier=4,
                 norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape',
                 b_rope_or_upe=10000, head_ratio_upe=2):
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
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)
        elif norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            num_heads, d_model, max_seq_len=max_seq_len, pe_type=pe_type,
            b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe,
        )
        if ffn_type == 'swiglu':
            self.ffn = SwiGLU(d_model, d_model * multiplier)
        elif ffn_type == 'gelu':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * multiplier),
                nn.GELU(),
                nn.Linear(d_model * multiplier, d_model),
            )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: 输出 [batch_size, seq_len, d_model]。
        """
        x_norm1 = self.norm1(x)
        h = self.attention(x_norm1)
        x1 = x + h
        x_norm2 = self.norm2(x1)
        x_out = self.ffn(x_norm2)
        x2 = x1 + x_out
        return x2
