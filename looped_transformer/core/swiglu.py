"""SwiGLU 门控前馈网络。

提供 SwiGLU 激活单元，与 PyTorch 自带的 GELU 并列，作为 TransformerBlock
的 FFN 选项之一。计算方式：W2(silu(W(x)) * V(x))。
"""

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈网络。

    使用 SiLU 激活的门控线性单元，相比 GELU 在部分任务上表现更好。

    Attributes:
        W (nn.Linear): 升维投影（门控分支）。
        V (nn.Linear): 升维投影（值分支）。
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
        x_out = self.W2(x1 * x2)
        return x_out
