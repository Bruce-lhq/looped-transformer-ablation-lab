"""注意力探针模块。

提供两个基于 PyTorch forward hook 的无感探针：
- AttentionProbe：捕获各层的完整注意力权重矩阵；
- SinkMetricsProbe：计算 sink score / sink rate，量化注意力向第 0 个位置集中的程度。

两者都作为 hook 注册到 MultiHeadAttention 上，读取其 captured_attention 属性。
"""

import torch


class AttentionProbe:
    """注意力探针：捕获完整注意力矩阵。

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
            self.captured_data.append(None)  # 如果没有捕获到注意力矩阵，记录一个None占位符

    def reset(self):
        """清空已捕获的数据，在每次前向传播前调用以避免混淆。"""
        self.captured_data = []


class SinkMetricsProbe:
    """Sink 指标探针：度量注意力的 sink 倾向。

    通过 hook 读取 captured_attention，计算两个指标：
    - sink score：所有头对第 0 个位置的平均注意力；
    - sink rate：sink score 超过 threshold 的头数占比。

    Attributes:
        threshold (float): 判断 sink 的阈值。
        captured_scores (list): 每次前向传播的 sink score。
        captured_rates (list): 每次前向传播的 sink rate。
    """

    def __init__(self, threshold=0.3):
        """初始化 SinkMetricsProbe。

        Args:
            threshold (float): sink score 的判断阈值，默认 0.3。
        """
        self.threshold = threshold
        self.captured_scores = []
        self.captured_rates = []

    def __call__(self, module, input, output):
        """Hook 回调函数。

        Args:
            module (nn.Module): 被 hook 的模块（MultiHeadAttention 实例）。
            input: 模型输入（未使用）。
            output: 模型输出（未使用）。
        """
        if getattr(module, 'captured_attention', None) is not None:
            with torch.no_grad():  # attention矩阵仍在GPU上
                attention = module.captured_attention  # [batch_size, num_heads, seq_len, seq_len]
                sink_attention = attention[:, :, :, 0]  # [batch_size, num_heads, seq_len]
                score_head = sink_attention.mean(dim=2).mean(dim=0)  # [num_heads]
                score = score_head.mean().item()  # 平均所有头的 sink score
                rate = (score_head >= self.threshold).float().mean().item()  # 大于等于 threshold 的头占总头数的比例
                self.captured_scores.append(score)
                self.captured_rates.append(rate)
        else:
            self.captured_scores.append(None)  # 如果没有捕获到注意力矩阵，记录一个None占位符
            self.captured_rates.append(None)

    def reset(self):
        """清空已捕获的指标，在每次前向传播前调用。"""
        self.captured_scores = []
        self.captured_rates = []
