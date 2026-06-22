"""任务级数据生成器。

提供三个并列的回归任务数据源：linear / nonlinear / lorenz，接口一致（返回
``(x_data, y_data)``）。三者都支持 ``ood_kwargs`` 做分布偏移，linear/nonlinear
支持 ``valid_d_x`` 控制有效输入维度（多余维度置零）。Lorenz 的底层动力学
（RK4、kernel、离线池化）见 ``core.lorenz``。

本模块还持有 lorenz 运行时缓存的线程锁 ``_lorenz_cache_lock`` 与清除函数
``clear_lorenz_cache``，二者服务于 ``lorenz_data_generator`` 的池化缓存加载。
"""

import os
import threading

import torch
import torch.nn.functional as F

from .core.lorenz import lorenz_kernel

# 保护 lorenz_data_generator 缓存加载的线程锁（多线程并行实验时避免重复 torch.load）
_lorenz_cache_lock = threading.Lock()


def linear_data_generator(batch_size, seq_len, valid_d_x=None, d_x=20, d_y=1, device='cpu', generator=None, ood_kwargs=None):
    """生成一批线性回归数据 ``y = x @ w``。

    每次调用随机采样真实权重 ``w ~ N(0, I/valid_d_x)`` 并生成 (x, y) 对。

    Args:
        batch_size (int): 批次大小。
        seq_len (int): 序列长度，上下文样本数 k = seq_len // 2。
        valid_d_x (int or None): 有效输入维度；小于 d_x 时多余维度权重与数据置零。
            None 表示 valid_d_x = d_x。
        d_x (int): 输入特征维度。
        d_y (int): 输出标签维度。
        device (str): 张量所在设备。
        generator (torch.Generator or None): 随机数生成器，用于可复现性。
        ood_kwargs (dict or None): OOD 配置，支持 x_distribution / x_scale /
            x_mean_shift / x_noise_std 与对应的 y_* 选项。

    Returns:
        tuple: (x_data [batch_size, k, d_x], y_data [batch_size, k, d_y])。
    """
    with torch.no_grad():
        if ood_kwargs is None:
            ood_kwargs = {}
        if valid_d_x is None:
            valid_d_x = d_x
        w = torch.randn(batch_size, d_x, d_y, device=device, generator=generator) / (valid_d_x ** 0.5)  # 真实线性函数 w: [batch_size, d_x, d_y]
        if valid_d_x < d_x:
            w[:, valid_d_x:, :] = 0.0  # 将多余的输入维度对应的权重置零，保证它们对输出没有影响
        x_distribution = ood_kwargs.get('x_distribution', 'gaussian')
        if x_distribution == 'gaussian':
            x_data = torch.randn(batch_size, seq_len//2, d_x, device=device, generator=generator)  # 输入特征 x: [batch_size, seq_len//2, d_x]
        elif x_distribution == 'uniform':
            x_data = (torch.rand(batch_size, seq_len//2, d_x, device=device, generator=generator) * 2 - 1) / (3**0.5)  # 均匀分布,方差归一化为1
        elif x_distribution == 'laplace':
            x_data = torch.distributions.Laplace(0, 1/(2**0.5)).sample((batch_size, seq_len//2, d_x)).to(device)  # 拉普拉斯分布，方差归一化为1

        if 'x_scale' in ood_kwargs:
            x_data = x_data * ood_kwargs['x_scale']  # 调整输入特征的尺度，制造OOD数据
        if 'x_mean_shift' in ood_kwargs:
            x_data = x_data + ood_kwargs['x_mean_shift']  # 平移输入特征的均值，制造OOD数据
        if 'x_noise_std' in ood_kwargs:
            x_data = x_data + torch.randn_like(x_data) * ood_kwargs['x_noise_std']  # 添加输入特征的噪声，制造OOD数据

        if valid_d_x < d_x:
            x_data[..., valid_d_x:] = 0.0  # 将多余的输入维度对应的数据置零，保证它们对输出没有影响

        y_data = x_data @ w  # 计算标签 y = x @ w: [batch_size, seq_len//2, d_y]

        if 'y_scale' in ood_kwargs:
            y_data = y_data * ood_kwargs['y_scale']  # 调整标签的尺度，制造OOD数据
        if 'y_mean_shift' in ood_kwargs:
            y_data = y_data + ood_kwargs['y_mean_shift']  # 平移标签的均值，制造OOD数据
        if 'y_noise_std' in ood_kwargs:
            y_data = y_data + torch.randn_like(y_data) * ood_kwargs['y_noise_std']  # 添加标签的噪声，制造OOD数据

        return x_data, y_data


def nonlinear_data_generator(batch_size, seq_len, valid_d_x=None, d_x=20, d_y=1, d_hidden=None, function_callable=None, device='cpu', generator=None, ood_kwargs=None):
    """生成一批非线性回归数据。

    数据生成形式：``linear(d_x->d_hidden) + nonlinear_func + linear(d_hidden->d_y)``。

    - ``d_hidden=d_x``：第一层线性变换退化为恒等映射，形式变为 nonlinear_func + linear；
    - ``d_hidden=d_y``：第二层线性变换退化为恒等映射，形式变为 linear + nonlinear_func。

    Args:
        batch_size (int): 批次大小。
        seq_len (int): 序列长度，k = seq_len // 2。
        valid_d_x (int or None): 有效输入维度。
        d_x (int): 输入特征维度。
        d_y (int): 输出标签维度。
        d_hidden (int): 隐藏层维度，控制非线性函数的输入输出维度。
        function_callable (callable): 非线性函数，如
            ``lambda x: F.relu(2*torch.sin(x)) + torch.cos(x)``。
        device (str): 张量所在设备。
        generator (torch.Generator or None): 随机数生成器。
        ood_kwargs (dict or None): OOD 配置；其中 ``function_callable`` 可在此覆盖
            以制造 Concept Shift，其余同 linear_data_generator。

    Returns:
        tuple: (x_data [batch_size, k, d_x], y_data [batch_size, k, d_y])。
    """
    '''
    linear(d_x->d_hidden) + nonlinear_func + linear(d_hidden->d_y) 的形式生成数据
    function_callable: 非线性函数, 如 lambda x: F.relu(2*torch.sin(x))+torch.cos(x)
    d_hidden: 隐藏层维度，控制非线性函数的输入输出维度。
              若d_hidden=d_x，则第一层线性变换默认设定为恒等映射，数据生成过程变为nonlinear_func + linear的形式；
              若d_hidden=d_y，则第二层线性变换默认设定为恒等映射，数据生成过程变为linear + nonlinear_func的形式。
    '''
    with torch.no_grad():
        if ood_kwargs is None:
            ood_kwargs = {}
        if valid_d_x is None:
            valid_d_x = d_x

        w1 = torch.randn(batch_size, d_x, d_hidden, device=device, generator=generator) / (valid_d_x ** 0.5)  # 第一层权重 w1: [batch_size, d_x, d_hidden]
        w2 = torch.randn(batch_size, d_hidden, d_y, device=device, generator=generator) / (d_hidden ** 0.5)  # 第二层权重 w2: [batch_size, d_hidden, d_y]
        if valid_d_x < d_x:
            w1[:, valid_d_x:, :] = 0.0  # 将多余的输入维度对应的权重置零，保证它们对输出没有影响
        x_distribution = ood_kwargs.get('x_distribution', 'gaussian')
        if x_distribution == 'gaussian':
            x_data = torch.randn(batch_size, seq_len//2, d_x, device=device, generator=generator)  # 输入特征 x: [batch_size, seq_len//2, d_x]
        elif x_distribution == 'uniform':
            x_data = (torch.rand(batch_size, seq_len//2, d_x, device=device, generator=generator) * 2 - 1) / (3**0.5)  # 均匀分布,方差归一化为1
        elif x_distribution == 'laplace':
            x_data = torch.distributions.Laplace(0, 1/(2**0.5)).sample((batch_size, seq_len//2, d_x)).to(device)  # 拉普拉斯分布，方差归一化为1

        if 'x_scale' in ood_kwargs:
            x_data = x_data * ood_kwargs['x_scale']  # 调整输入特征的尺度，制造OOD数据
        if 'x_mean_shift' in ood_kwargs:
            x_data = x_data + ood_kwargs['x_mean_shift']  # 平移输入特征的均值，制造OOD数据
        if 'x_noise_std' in ood_kwargs:
            x_data = x_data + torch.randn_like(x_data) * ood_kwargs['x_noise_std']  # 添加输入特征的噪声，制造OOD数据

        if valid_d_x < d_x:
            x_data[..., valid_d_x:] = 0.0  # 将多余的输入维度对应的数据置零，保证它们对输出没有影响

        if d_x == d_hidden:
            w1 = torch.eye(d_x, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # 如果输入维度和隐藏层维度相同，变为nonlinear_func + linear的形式
            if not getattr(nonlinear_data_generator, '_has_warned_w1_identity', False):
                print("Warning: w1 is set to identity matrix because d_x equals d_hidden.")
                nonlinear_data_generator._has_warned_w1_identity = True
        if d_hidden == d_y:
            w2 = torch.eye(d_hidden, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # 如果隐藏层维度和输出维度相同，变为linear + nonlinear_func的形式
            if not getattr(nonlinear_data_generator, '_has_warned_w2_identity', False):
                print("Warning: w2 is set to identity matrix because d_hidden equals d_y.")
                nonlinear_data_generator._has_warned_w2_identity = True

        hidden = x_data @ w1  # [batch_size, seq_len//2, d_hidden]

        func = ood_kwargs.get('function_callable', function_callable)  # Concept Shift
        # 应用非线性函数
        hidden = func(hidden)  # [batch_size, seq_len//2, d_hidden]
        # 第二层线性变换
        y_data = hidden @ w2  # [batch_size, seq_len//2, d_y]

        if 'y_scale' in ood_kwargs:
            y_data = y_data * ood_kwargs['y_scale']  # 调整标签的尺度，制造OOD数据
        if 'y_mean_shift' in ood_kwargs:
            y_data = y_data + ood_kwargs['y_mean_shift']  # 平移标签的均值，制造OOD数据
        if 'y_noise_std' in ood_kwargs:
            y_data = y_data + torch.randn_like(y_data) * ood_kwargs['y_noise_std']  # 添加标签的噪声，制造OOD数据

        return x_data, y_data


def lorenz_data_generator(batch_size, seq_len, valid_d_x=None, d_x=3, d_y=3, lorenz_kwargs=None, device='cpu', generator=None, ood_kwargs=None, load_path=None):
    """生成一批 Lorenz 吸引子序列数据。

    支持两种来源：
    - ``load_path``：从离线池（``create_lorenz_pool`` 产物）随机抽取轨迹，首次加载
      后缓存在函数对象上（线程安全，由 ``_lorenz_cache_lock`` 保护）；
    - None：用 ``lorenz_kernel`` 实时积分生成，支持 OOD（初始分布、噪声、参数偏移）。

    数据按"下一帧预测"组织：x = trajectory[:-1]，y = trajectory[1:]（错位一帧）。

    Args:
        batch_size (int): 批次大小。
        seq_len (int): 序列长度，k = seq_len // 2。
        valid_d_x (int or None): 有效输入维度（对 lorenz 默认 3）。
        d_x (int): 输入特征维度，默认 3。
        d_y (int): 输出标签维度，默认 3。
        lorenz_kwargs (dict or None): ``burn_in``（预热步数，默认 500）、``dt``（默认 0.01）。
        device (str): 张量所在设备。
        generator (torch.Generator or None): 随机数生成器（仅实时生成时使用）。
        ood_kwargs (dict or None): OOD 配置——init_distribution
            (gaussian/uniform/laplace)、lorenz_noise_std、sigma_shift/rho_shift/beta_shift。
        load_path (str or None): 离线池路径；None 表示实时生成。

    Returns:
        tuple: (x_data [batch_size, k, d_x], y_data [batch_size, k, d_y])。
    """
    with torch.no_grad():  # 数据生成过程中不需要计算梯度，使用no_grad加速
        if ood_kwargs is None:
            ood_kwargs = {}
        if valid_d_x is None:
            valid_d_x = d_x
        if lorenz_kwargs is None:
            lorenz_kwargs = {}
        burn_in = lorenz_kwargs.get('burn_in', 500)  # 预热步数，默认为500步，确保系统状态进入混沌吸引子
        if load_path is not None and not os.path.exists(load_path):
            print(f"Warning: load_path {load_path} does not exist. Generating new Lorenz data.")
            load_path = None
        if load_path is not None:
            with _lorenz_cache_lock:
                if not hasattr(lorenz_data_generator, "_cache") or lorenz_data_generator._cache_path != load_path:
                    print(f"Loading Lorenz data from {load_path}...")
                    checkpoint = torch.load(load_path, map_location='cpu')
                    lorenz_data_generator._cache = checkpoint['trajectory']
                    lorenz_data_generator._cache_path = load_path
                    if ood_kwargs.get('init_distribution', 'gaussian') != 'gaussian' or ood_kwargs.get('lorenz_noise_std', 0.0) > 0.0 or ood_kwargs.get('sigma_shift', 0.0) != 0.0 or ood_kwargs.get('rho_shift', 0.0) != 0.0 or ood_kwargs.get('beta_shift', 0.0) != 0.0:
                        print("Warning: The loaded Lorenz data was generated with a Gaussian distribution , no noise and default parameters. If you want to generate OOD data with different distribution or noise, please set load_path to None.")
            pool = lorenz_data_generator._cache  # [pool_size, traj_len, 3]
            pool_size = pool.shape[0]
            indices = torch.randint(0, pool_size, (batch_size,), device='cpu')
            batch_traj = pool[indices]  # [batch_size, traj_len, 3]
            if pool.shape[1] < burn_in + seq_len//2 + 1:
                old_burn_in = burn_in
                burn_in = max(0, pool.shape[1] - seq_len//2 - 1)
                print(f"Warning: The trajectories in the pool have length {pool.shape[1]}, which is less than burn_in + seq_len//2 + 1 = {old_burn_in + seq_len//2 + 1}. Burn-in will be automatically reduced to {burn_in}.")
            batch_traj = batch_traj[:, burn_in:burn_in+seq_len//2+1, :].to(device)  # [batch_size, seq_len//2+1, 3]
            x_data = batch_traj[:, :seq_len//2, :]  # [batch_size, seq_len//2, 3]
            y_data = batch_traj[:, 1:seq_len//2+1, :]  # [batch_size, seq_len//2, 3]

        else:
            dt = lorenz_kwargs.get('dt', 0.01)  # 时间步长默认为0.01
            # 初始化
            total_steps = burn_in + seq_len//2 + 1
            init_distribution = ood_kwargs.get('init_distribution', 'gaussian')
            if init_distribution == 'gaussian':
                init_state = torch.randn(batch_size, 3, device=device, generator=generator) * 15 + torch.tensor([0, 0, 25.0], device=device)  # 初始状态（放大15倍并平移(0, 0, 25)以确保在洛伦兹吸引子中心附近）
            elif init_distribution == 'uniform':
                init_state = (torch.rand(batch_size, 3, device=device, generator=generator) * 30 - 15) + torch.tensor([0, 0, 25.0], device=device)  # 均匀分布，范围[-15, 15]，并平移(0, 0, 25)
            elif init_distribution == 'laplace':
                u = torch.rand(batch_size, 3, device=device, generator=generator) - 0.5
                init_state = torch.tensor([0, 0, 25.0], device=device) - (15 / (2**0.5)) * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
            noise_std = ood_kwargs.get('lorenz_noise_std', 0.0)
            sigma = ood_kwargs.get('sigma_shift', 0.0) + 10.0
            rho = ood_kwargs.get('rho_shift', 0.0) + 28.0
            beta = ood_kwargs.get('beta_shift', 0.0) + 8.0/3.0
            # 生成轨迹
            trajectory = lorenz_kernel(init_state=init_state, total_steps=total_steps, dt=dt, sigma=sigma, rho=rho, beta=beta)[:, burn_in:, :]  # [batch_size, seq_len//2+1, 3]
            if noise_std > 0.0:
                trajectory = trajectory + torch.randn_like(trajectory) * noise_std  # 在观测轨迹上添加噪声，制造OOD数据
            x_data = trajectory[:, :-1, :]  # x_data: [batch_size, seq_len//2, 3]
            y_data = trajectory[:, 1:, :]  # y_data: [batch_size, seq_len//2, 3]

        # 裁剪或填充至给定维度
        if valid_d_x > 3:
            x_data = F.pad(x_data, (0, valid_d_x - 3), value=0.0)  # 将输入特征维度扩展到 valid_d_x，新增的维度填充为0
        elif valid_d_x < 3:
            x_data = x_data[:, :, :valid_d_x]  # 将输入特征维度裁剪到 valid_d_x
        if d_y > 3:
            y_data = F.pad(y_data, (0, d_y - 3), value=0.0)  # 将标签维度扩展到 d_y，新增的维度填充为0
        elif d_y < 3:
            y_data = y_data[:, :, :d_y]  # 将标签维度裁剪到 d_y
        return x_data, y_data


def clear_lorenz_cache():
    """清除 ``lorenz_data_generator`` 上缓存的离线池。

    缓存挂在 ``lorenz_data_generator`` 函数对象的 ``_cache`` / ``_cache_path`` 属性上。
    无缓存时打印提示。
    """
    if hasattr(lorenz_data_generator, "_cache"):
        del lorenz_data_generator._cache
        del lorenz_data_generator._cache_path
        print("Lorenz data cache cleared.")
    else:
        print("No Lorenz data cache found.")
