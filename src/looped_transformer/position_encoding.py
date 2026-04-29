"""位置编码模块。

提供五种位置编码实现：APE、LearnedAPE、ALiBi、RoPE、MS_UPE。
"""

import torch
import torch.nn as nn


class APE(nn.Module):
    """绝对位置编码 (Absolute Position Encoding)。

    使用正弦/余弦函数生成固定的位置编码矩阵，直接加到输入上。
    偶数维度使用 sin，奇数维度使用 cos。

    Attributes:
        pe (torch.Tensor): 预计算的位置编码矩阵，形状 [1, max_seq_len, d_model]。
    """

    def __init__(self, d_model, max_seq_len=4096, b=10000):
        """初始化 APE。

        Args:
            d_model (int): 模型隐藏层维度。
            max_seq_len (int): 最大序列长度。
            b (int): 频率基数，默认 10000。
        """
        super().__init__()
        theta_i = 1 / (b ** (torch.arange(0, d_model, 2).float() / d_model))
        m = torch.arange(max_seq_len).float()
        m_theta_i = torch.outer(m, theta_i)
        stacked = torch.stack((torch.sin(m_theta_i), torch.cos(m_theta_i)), dim=-1)
        pe = stacked.flatten(1, 2)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """前向传播：将位置编码加到输入上。

        Args:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: x + pe，形状与输入相同。
        """
        return x + self.pe[:, :x.shape[1], :]


class LearnedAPE(nn.Module):
    """可学习绝对位置编码 (Learned Absolute Position Encoding)。

    使用 nn.Embedding 将位置索引映射为可学习的向量，直接加到输入上。

    Attributes:
        pe (nn.Embedding): 可学习的位置 embedding [max_seq_len, d_model]。
    """

    def __init__(self, d_model, max_seq_len=4096):
        """初始化 LearnedAPE。

        Args:
            d_model (int): 模型隐藏层维度。
            max_seq_len (int): 最大序列长度。
        """
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """前向传播：将可学习位置编码加到输入上。

        Args:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, d_model]。

        Returns:
            torch.Tensor: x + pe，形状与输入相同。
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class ALiBi(nn.Module):
    """注意力线性偏置 (Attention with Linear Biases)。

    在注意力分数矩阵上施加与距离成正比的线性惩罚项。
    slope 由头数 H 硬编码决定：m_h = 2^{-8h/H}。
    位置编码不加在输入端，而是通过修改 score = QK^T - m * |i-j| 实现。
    同时集成了因果掩码（下三角保留，上三角置 -inf）。

    Attributes:
        fused_mask (torch.Tensor): 预计算的偏置 + 因果掩码，形状 [num_heads, max_seq_len, max_seq_len]。
    """

    def __init__(self, num_heads, max_seq_len=4096):
        """初始化 ALiBi。

        Args:
            num_heads (int): 注意力头数。
            max_seq_len (int): 最大序列长度。
        """
        super().__init__()
        dist = torch.arange(max_seq_len)
        dist_matrix = torch.abs(dist.view(-1, 1) - dist.view(1, -1))
        bias = -(2 ** -(8 * torch.arange(1, num_heads + 1) / num_heads)).view(-1, 1, 1) * dist_matrix.unsqueeze(0)
        tril = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        mask = torch.zeros(max_seq_len, max_seq_len).masked_fill(~tril, float('-inf'))
        self.register_buffer('fused_mask', bias + mask)

    def forward(self, seq_len):
        """返回指定序列长度的偏置掩码。

        Args:
            seq_len (int): 当前序列长度。

        Returns:
            torch.Tensor: 偏置掩码 [num_heads, seq_len, seq_len]。
        """
        return self.fused_mask[:, :seq_len, :seq_len]


class RoPE(nn.Module):
    """旋转位置编码 (Rotary Position Encoding)。

    在 MHA 内部对 Q 和 K 的特征维度施加旋转变换，不改变 V。
    通过复数域的旋转实现相对位置信息的注入。

    Attributes:
        cos (torch.Tensor): 预计算的 cos 值 [1, 1, max_seq_len, d_k]。
        sin (torch.Tensor): 预计算的 sin 值 [1, 1, max_seq_len, d_k]。
    """

    def __init__(self, d_k, max_seq_len=4096, b=10000):
        """初始化 RoPE。

        Args:
            d_k (int): 每个注意力头的维度。
            max_seq_len (int): 最大序列长度。
            b (int): 频率基数，默认 10000。
        """
        super().__init__()
        theta_i = 1 / (b ** (torch.arange(0, d_k, 2).float() / d_k))
        m = torch.arange(max_seq_len).float()
        m_theta_i = torch.outer(m, theta_i)
        cos = torch.cos(torch.cat((m_theta_i, m_theta_i), dim=-1))
        sin = torch.sin(torch.cat((m_theta_i, m_theta_i), dim=-1))
        self.register_buffer('cos', cos[None, None, :, :])
        self.register_buffer('sin', sin[None, None, :, :])

    def forward(self, x):
        """前向传播：对 Q 或 K 施加旋转变换。

        Args:
            x (torch.Tensor): Q 或 K 张量 [batch_size, num_heads, seq_len, d_k]。

        Returns:
            torch.Tensor: 旋转后的张量，形状与输入相同。
        """
        seq_len = x.shape[2]
        d_2 = x.shape[-1] // 2
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        x_first_half = x[..., :d_2]
        x_second_half = x[..., d_2:]
        x_flip = torch.cat((-x_second_half, x_first_half), dim=-1)
        return x * cos + x_flip * sin


class MS_UPE(nn.Module):
    """多尺度解绑位置编码 (Multi-Scale Untied Position Encoding)。

    自创的 PE 方法，与 RoPE 类似在 Q/K 上操作，但使用加法（而非旋转）注入位置信息。
    每个注意力头有独立的基频 b_h = b_0 * head_ratio^h，实现多尺度解绑。
    相比 RoPE 计算更快（仅加法，无旋转操作）。

    Attributes:
        pe (torch.Tensor): 预计算的位置编码 [1, num_heads, max_seq_len, d_k]。
    """

    def __init__(self, num_heads, d_k, max_seq_len=4096, b_0=10000, head_ratio=2):
        """初始化 MS_UPE。

        Args:
            num_heads (int): 注意力头数。
            d_k (int): 每个注意力头的维度。
            max_seq_len (int): 最大序列长度。
            b_0 (int): 第 0 头的基频，默认 10000。
            head_ratio (int): 相邻头的基频倍率，默认 2。
        """
        super().__init__()
        b = (b_0 * (head_ratio ** (torch.arange(0, num_heads)))).float()
        theta_i = 1 / (b.unsqueeze(1) ** (torch.arange(0, d_k, 2).float() / d_k).unsqueeze(0))
        m = torch.arange(max_seq_len).float()
        m_theta_i = torch.einsum('m,hk->hmk', m, theta_i)
        pe = torch.cat((torch.cos(m_theta_i), torch.sin(m_theta_i)), dim=-1)
        self.register_buffer('pe', pe[None, :, :, :])

    def forward(self, x):
        """前向传播：将多尺度位置编码加到输入上。

        Args:
            x (torch.Tensor): Q 或 K 张量 [batch_size, num_heads, seq_len, d_k]。

        Returns:
            torch.Tensor: x + pe，形状与输入相同。
        """
        seq_len = x.shape[2]
        pe = self.pe[:, :, :seq_len, :]
        return x + pe
