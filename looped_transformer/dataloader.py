"""统一数据加载器。

按 ``data_type`` 将请求分发到 ``data_generators`` 中对应的生成器，并按需在序列前端
插入 sink padding（全零 token）。返回一个永不耗尽的生成器。
"""

import torch.nn.functional as F

from .data_generators import linear_data_generator, nonlinear_data_generator, lorenz_data_generator


def dataloader(batch_size=64, seq_len=80, valid_d_x=None, d_x=20, d_y=1, device='cpu',
               data_type='linear', d_hidden=None, function_callable=None, sink_padding=1,
               generator=None, ood_kwargs=None, lorenz_kwargs=None, load_lorenz_from=None):
    """无限批量数据加载器（生成器）。

    持续 yield 新的数据批次，永不耗尽。按 ``data_type`` 调用对应的 generator，
    可选在序列前端插入 ``sink_padding`` 组全零 (x, y) token。

    Args:
        batch_size (int): 批次大小。
        seq_len (int): 序列长度。
        valid_d_x (int or None): 有效输入维度，透传给 generator。
        d_x (int): 输入特征维度。
        d_y (int): 输出标签维度。
        device (str): 张量所在设备。
        data_type (str): 数据类型，'linear' / 'nonlinear' / 'lorenz'。
        d_hidden (int or None): nonlinear 的隐藏层维度。
        function_callable (callable or None): nonlinear 的非线性函数。
        sink_padding (int or None): 序列前端填充的 (x, y) 全零 token 组数；
            None 表示不填充。
        generator (torch.Generator or None): 随机数生成器。
        ood_kwargs (dict or None): OOD 配置，透传给 generator。
        lorenz_kwargs (dict or None): lorenz 专用参数（burn_in / dt）。
        load_lorenz_from (str or None): lorenz 离线池路径。

    Yields:
        tuple: (x_data, y_data) 每个批次的数据。
    """
    if valid_d_x is None:
        valid_d_x = d_x
    while True:
        if data_type == 'linear':
            x_data, y_data = linear_data_generator(batch_size=batch_size, seq_len=seq_len, valid_d_x=valid_d_x, d_x=d_x, d_y=d_y, device=device, generator=generator, ood_kwargs=ood_kwargs)
        elif data_type == 'nonlinear':
            x_data, y_data = nonlinear_data_generator(batch_size=batch_size, seq_len=seq_len, valid_d_x=valid_d_x, d_x=d_x, d_y=d_y, d_hidden=d_hidden, function_callable=function_callable, device=device, generator=generator, ood_kwargs=ood_kwargs)
        elif data_type == 'lorenz':
            x_data, y_data = lorenz_data_generator(batch_size=batch_size, seq_len=seq_len, valid_d_x=valid_d_x, d_x=d_x, d_y=d_y, device=device, generator=generator, ood_kwargs=ood_kwargs, lorenz_kwargs=lorenz_kwargs, load_path=load_lorenz_from)
        if sink_padding is not None:
            x_data = F.pad(x_data, (0, 0, sink_padding, 0), value=0)  # 在seq_len维度最前面添加sink_padding个全0向量
            y_data = F.pad(y_data, (0, 0, sink_padding, 0), value=0)
        yield x_data, y_data
