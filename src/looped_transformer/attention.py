"""多头注意力模块。

支持多种位置编码（RoPE、MS_UPE、ALiBi、APE、LearnedAPE）的可配置 Multi-Head Attention。
训练时使用 fused scaled_dot_product_attention，推理时记录原始注意力矩阵供后续分析。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import RoPE, MS_UPE, ALiBi


class MultiHeadAttention(nn.Module):
    """可配置位置编码的多头注意力。

    根据传入的 pe_type 列表自动分发位置编码：RoPE/MS_UPE 作用于 Q/K，
    ALiBi 作用于 score 矩阵，APE/LearnedAPE 由外层 ToyModel 在输入端处理。

    推理模式（eval）使用手动实现的 scaled dot-product attention，并将注意力矩阵
    写入 self.captured_attention，供 AttentionProbe 通过 hook 捕获。
    训练模式使用 PyTorch 的 F.scaled_dot_product_attention（fused kernel）。

    Attributes:
        W_Q, W_K, W_V, W_O (nn.Linear): QKV 投影和输出投影矩阵。
        qk_pe_modules (nn.ModuleList): 作用于 Q/K 的位置编码模块列表。
        sc_pe_modules (nn.ModuleList): 作用于 score 矩阵的位置编码模块列表。
        mask (torch.Tensor): 因果掩码 [1, 1, max_seq_len, max_seq_len]。
        captured_attention (torch.Tensor or None): 最近一次推理的注意力权重。
    """

    def __init__(self, num_heads, d_model, max_seq_len=4096, pe_type='learned_ape',
                 b_rope_or_upe=10000, head_ratio_upe=2):
        """初始化 MultiHeadAttention。

        Args:
            num_heads (int): 注意力头数 H。
            d_model (int): 模型总维度 D，必须能被 num_heads 整除。
            max_seq_len (int): 最大序列长度，用于预分配掩码和位置编码。
            pe_type (str or list[str]): 位置编码类型，可选 'ape', 'learned_ape',
                'rope', 'ms_upe', 'alibi'。传入列表可叠加多种 PE。
            b_rope_or_upe (int): RoPE 或 MS_UPE 的基频参数。
            head_ratio_upe (int): MS_UPE 相邻头的基频倍率。

        Raises:
            AssertionError: 当 d_model 不能被 num_heads 整除时抛出。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.pe_type = [pe_type] if isinstance(pe_type, str) else pe_type
        self.qk_pe_modules = nn.ModuleList()
        self.sc_pe_modules = nn.ModuleList()
        for pe in self.pe_type:
            if pe == 'rope':
                pe_module = RoPE(self.d_k, max_seq_len, b=b_rope_or_upe)
                self.qk_pe_modules.append(pe_module)
            elif pe == 'ms_upe':
                pe_module = MS_UPE(num_heads, self.d_k, max_seq_len,
                                   b_0=b_rope_or_upe, head_ratio=head_ratio_upe)
                self.qk_pe_modules.append(pe_module)
            elif pe == 'alibi':
                pe_module = ALiBi(num_heads, max_seq_len)
                self.sc_pe_modules.append(pe_module)
            elif pe in ['ape', 'learned_ape']:
                pass

        tril = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        mask = torch.zeros(max_seq_len, max_seq_len).masked_fill(~tril, float('-inf'))
        self.register_buffer('mask', mask[None, None, :, :])

    def forward(self, x):
        """前向传播。

        训练模式使用 fused SDPA（更快），推理模式手动计算以捕获注意力矩阵。

        Args:
            x (torch.Tensor): 输入 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: MHA 输出 [batch_size, seq_len, d_model]。
        """
        batch_size, seq_len, _ = x.shape
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        for pe_module in self.qk_pe_modules:
            Q = pe_module(Q)
            K = pe_module(K)

        if not self.training:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            if self.sc_pe_modules:
                fused_mask = self.sc_pe_modules[0](seq_len)
                scores = scores + fused_mask
                attention = F.softmax(scores, dim=-1)
            else:
                attention = F.softmax(scores + self.mask[..., :seq_len, :seq_len], dim=-1)
            self.captured_attention = attention
            out = torch.matmul(attention, V)
        else:
            if self.sc_pe_modules:
                fused_mask = self.sc_pe_modules[0](seq_len)
                out = F.scaled_dot_product_attention(Q, K, V, attn_mask=fused_mask, is_causal=False)
            else:
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        H = self.W_O(out)
        return H
