"""数据流水线模块。

提供线性回归数据生成器和无限批量数据加载器，支持 sink padding。
"""

import torch
import torch.nn.functional as F


def linear_data_generator(batch_size, seq_len, d_x, d_y, device='cpu'):
    """生成一批线性回归数据。

    每次调用随机采样真实权重 w ~ N(0, I) 并生成 y = x @ w 的 (x, y) 对。

    Args:
        batch_size (int): 批次大小。
        seq_len (int): 序列长度，其中的一半为上下文样本数 k = seq_len // 2。
        d_x (int): 输入特征 x 的维度。
        d_y (int): 输出标签 y 的维度。
        device (str): 张量所在设备，默认 'cpu'。

    Returns:
        tuple: (x_data, y_data)
            - x_data [batch_size, k, d_x]
            - y_data [batch_size, k, d_y]
    """
    w = torch.randn(batch_size, d_x, d_y, device=device)
    x_data = torch.randn(batch_size, seq_len // 2, d_x, device=device)
    y_data = x_data @ w
    return x_data, y_data


def dataloader(batch_size=64, seq_len=80, d_x=20, d_y=1, device='cpu',
               data_type='linear', sink_padding=1):
    """无限批量数据加载器（生成器）。

    持续 yield 新的数据批次，永不耗尽。支持在序列前端插入 sink token（全零向量）。

    Args:
        batch_size (int): 批次大小。
        seq_len (int): 序列长度。
        d_x (int): 输入特征维度。
        d_y (int): 输出标签维度。
        device (str): 张量所在设备。
        data_type (str): 数据类型，目前仅支持 'linear'。
        sink_padding (int or None): 序列前端填充的 (x, y) 全零 token 组数。
            None 表示不填充。

    Yields:
        tuple: (x_data, y_data) 每个批次的数据。
    """
    while True:
        if data_type == 'linear':
            x_data, y_data = linear_data_generator(batch_size, seq_len, d_x, d_y, device=device)
        if sink_padding is not None:
            x_data = F.pad(x_data, (0, 0, sink_padding, 0), value=0)
            y_data = F.pad(y_data, (0, 0, sink_padding, 0), value=0)
        yield x_data, y_data
