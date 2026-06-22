"""多头注意力模块。

可配置多种位置编码（RoPE、MS_UPE、ALiBi、APE、LearnedAPE）的 Multi-Head Attention。
训练时使用 fused ``scaled_dot_product_attention``，推理时手动计算以捕获完整注意力矩阵
供 AttentionProbe / SinkMetricsProbe 通过 hook 分析。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.position_encoding import RoPE, MS_UPE, ALiBi


class MultiHeadAttention(nn.Module):
    """可配置位置编码的多头注意力。

    根据传入的 ``pe_type`` 列表自动分发位置编码：RoPE / MS_UPE 作用于 Q/K，
    ALiBi 作用于 score 矩阵，APE / LearnedAPE 由外层 ToyModel 在输入端处理。

    推理模式（eval）手动实现 scaled dot-product attention 并把注意力矩阵写入
    ``self.captured_attention``，供 probe 通过 hook 捕获；训练模式使用 PyTorch 的
    fused ``F.scaled_dot_product_attention`` 以提速，且不捕获注意力矩阵。

    Attributes:
        W_Q, W_K, W_V, W_O (nn.Linear): QKV 投影与输出投影。
        qk_pe_modules (nn.ModuleList): 作用于 Q/K 的位置编码模块（RoPE/MS_UPE）。
        sc_pe_modules (nn.ModuleList): 作用于 score 矩阵的位置编码模块（ALiBi）。
        mask (torch.Tensor): 因果掩码 [1, 1, max_seq_len, max_seq_len]。
        captured_attention (torch.Tensor or None): 最近一次推理的注意力权重。
    """

    def __init__(self, num_heads, d_model, max_seq_len=4096, pe_type='learned_ape', b_rope_or_upe=10000, head_ratio_upe=2):
        """初始化 MultiHeadAttention。

        Args:
            num_heads (int): 注意力头数 H。
            d_model (int): 模型总维度 D，必须能被 num_heads 整除。
            max_seq_len (int): 最大序列长度，用于预分配掩码与位置编码。
            pe_type (str or list[str]): 位置编码类型，可选 'ape'、'learned_ape'、
                'rope'、'ms_upe'、'alibi'；传列表可叠加多种 PE。
            b_rope_or_upe (int): RoPE / MS_UPE 的基频参数。
            head_ratio_upe (int): MS_UPE 相邻头的基频倍率。

        Raises:
            AssertionError: 当 d_model 不能被 num_heads 整除时。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # 维度属性
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        # 权重矩阵
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        # 位置编码模块
        self.pe_type = [pe_type] if isinstance(pe_type, str) else pe_type
        self.qk_pe_modules = nn.ModuleList()
        self.sc_pe_modules = nn.ModuleList()
        for pe in self.pe_type:
            if pe == 'rope':
                pe_module = RoPE(d_k=self.d_k, max_seq_len=max_seq_len, b=b_rope_or_upe)
                self.qk_pe_modules.append(pe_module)  # RoPE应用于Q和K
            elif pe == 'ms_upe':
                pe_module = MS_UPE(num_heads=num_heads, d_k=self.d_k, max_seq_len=max_seq_len, b_0=b_rope_or_upe, head_ratio=head_ratio_upe)
                self.qk_pe_modules.append(pe_module)  # MS-UPE应用于Q和K
            elif pe == 'alibi':
                pe_module = ALiBi(num_heads=num_heads, max_seq_len=max_seq_len)
                self.sc_pe_modules.append(pe_module)  # ALiBi应用于score矩阵
            elif pe in ['ape', 'learned_ape']:
                pass  # APE和Learned APE在输入时直接加到x上，不需要单独模块
        # 因果掩码
        tril = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()  # torch.tril (triangular lower)提取下三角元素，并把上三角元素置0
        mask = torch.zeros(max_seq_len, max_seq_len).masked_fill(~tril, float('-inf'))  # 下三角元素置0，上三角元素置-inf
        self.register_buffer('mask', mask[None, None, :, :])  # [1, 1, seq_len, seq_len]

    def forward(self, x):
        """前向传播。

        训练模式使用 fused SDPA（更快），推理模式手动计算以捕获注意力矩阵。

        Args:
            x (torch.Tensor): 输入 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: MHA 输出 [batch_size, seq_len, d_model]。
        """
        # x: [batch_size, seq_len, d_model]
        # 线性投影与分头
        batch_size, seq_len, _ = x.shape
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 交换(1, 2)是因为之后softmax时要进行矩阵乘法(只乘后两维)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 在seq_len维度上分配注意力
        # 根据选择的PE类型对Q和K进行位置编码
        for pe_module in self.qk_pe_modules:
            Q = pe_module(Q)
            K = pe_module(K)
        if not self.training:
            # 缩放点积注意力与因果掩码 Causal Masking
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
            if self.sc_pe_modules:
                fused_mask = self.sc_pe_modules[0](seq_len)  # [num_heads, seq_len, seq_len]
                scores = scores + fused_mask  # 将ALiBi的偏置加到得分上
                attention = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
            else:
                attention = F.softmax(scores + self.mask[..., :seq_len, :seq_len], dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
            self.captured_attention = attention  # 捕获注意力矩阵，用hook记录下来，方便后续分析
            out = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, d_k]
        else:
            self.captured_attention = None  # 训练模式下不捕获注意力矩阵，节省内存
            if self.sc_pe_modules:
                fused_mask = self.sc_pe_modules[0](seq_len)  # [num_heads, seq_len, seq_len]
                out = F.scaled_dot_product_attention(Q, K, V, attn_mask=fused_mask, is_causal=False)  # [batch_size, num_heads, seq_len, d_k]
            else:
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # [batch_size, num_heads, seq_len, d_k]
        # 多头拼接并乘以输出矩阵(必须先内存连续化(contiguous)再view，否则会报错)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, d_model]
        H = self.W_O(out)  # [batch_size, seq_len, d_model]
        return H
