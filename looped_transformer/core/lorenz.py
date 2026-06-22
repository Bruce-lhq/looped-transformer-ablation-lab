"""Lorenz 系统底层数学工具。

提供 Lorenz 吸引子的导数计算、RK4 数值积分、完整轨迹生成（kernel）以及
离线数据池的预生成与增量拼接。本模块只做纯数学计算与离线池化，不涉及
训练时的 batch 采样——后者见 ``data_generators.lorenz_data_generator``。
"""

import os

import torch


def lorenz_derivative(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz 系统的导数（右端项）。

    Args:
        state (torch.Tensor): 当前状态 [batch_size, 3]。
        sigma (float): Prandtl 数，默认 10.0。
        rho (float): Rayleigh 数，默认 28.0。
        beta (float): 几何参数，默认 8/3。

    Returns:
        torch.Tensor: 各维度的导数 [batch_size, 3]。
    """
    # state: [batch_size, 3]
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack((dx, dy, dz), dim=1)  # [batch_size, 3]


def rk4_step(state, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """用四阶 Runge-Kutta 法推进一个时间步。

    Args:
        state (torch.Tensor): 当前状态 [batch_size, 3]。
        dt (float): 时间步长，默认 0.01。
        sigma (float): Prandtl 数。
        rho (float): Rayleigh 数。
        beta (float): 几何参数。

    Returns:
        torch.Tensor: 推进后的状态 [batch_size, 3]。
    """
    k1 = lorenz_derivative(state, sigma, rho, beta)
    k2 = lorenz_derivative(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_derivative(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_derivative(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # 可选：torch.compile 加速
    # try:
    #     rk4_step = torch.compile(rk4_step)
    # except Exception:
    #     pass


def lorenz_kernel(init_state, total_steps, dt, sigma, rho, beta):
    """从初始状态出发，用 RK4 迭代 total_steps 步，返回完整轨迹。

    Args:
        init_state (torch.Tensor): 初始状态 [batch_size, 3]。
        total_steps (int): 迭代步数。
        dt (float): 时间步长。
        sigma (float): Prandtl 数。
        rho (float): Rayleigh 数。
        beta (float): 几何参数。

    Returns:
        torch.Tensor: 轨迹 [batch_size, total_steps, 3]。
    """
    # init_state: [batch_size, 3]
    batch_size = init_state.shape[0]
    device = init_state.device
    dtype = init_state.dtype
    trajectory = torch.empty((total_steps, batch_size, 3), device=device, dtype=dtype)
    current_state = init_state
    for t in range(total_steps):
        current_state = rk4_step(state=current_state, dt=dt, sigma=sigma, rho=rho, beta=beta)
        trajectory[t] = current_state
    trajectory = trajectory.transpose(0, 1)  # 转置为 [batch_size, total_steps, 3]
    return trajectory  # [batch_size, total_steps, 3]
    # 可选：torch.compile 加速
    # try:
    #     lorenz_kernel = torch.compile(lorenz_kernel)
    # except Exception:
    #     pass


def create_lorenz_pool(pool_size=10000, traj_len=500, dt=0.01, sigma=10.0, beta=8/3, rho=28.0, save_path='auto'):
    """预生成一批 Lorenz 轨迹并保存为离线数据池（支持增量拼接）。

    自动选择可用设备（CUDA > MPS > CPU）生成轨迹，移回 CPU 后保存。
    若 save_path 已存在且 traj_len 一致，则在 batch 维度增量拼接。

    Args:
        pool_size (int): 本次新生成的轨迹条数。
        traj_len (int): 每条轨迹的时序长度。
        dt (float): 时间步长。
        sigma (float): Prandtl 数。
        beta (float): 几何参数。
        rho (float): Rayleigh 数。
        save_path (str): 保存路径；'auto' 自动按参数命名，空串则不保存。

    Raises:
        ValueError: 增量拼接时新旧轨迹的 traj_len 不一致。
    """
    if save_path == 'auto':
        save_path = f'data/lorenz/length_{traj_len}_dt{dt}_sigma{sigma:.1f}_beta{beta:.1f}_rho{rho:.1f}.pth'
    with torch.no_grad():
        # 初始化
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        generator = torch.Generator(device=device)
        init_state = torch.randn(pool_size, 3, device=device, generator=generator) * 15 + torch.tensor([0, 0, 25], device=device)  # 初始状态（放大15倍并平移(0, 0, 25)以确保在洛伦兹吸引子中心附近）
        # 生成轨迹
        trajectory = lorenz_kernel(init_state=init_state, total_steps=traj_len, dt=dt, sigma=sigma, rho=rho, beta=beta)  # [pool_size, traj_len, 3]
        trajectory = trajectory.cpu()  # 将数据移回CPU，准备保存

        if save_path and os.path.exists(save_path):
            print(f"检测到已有数据文件，正在读取并进行增量拼接...")
            old_data = torch.load(save_path, map_location='cpu')
            old_trajectory = old_data['trajectory']
            if old_trajectory.shape[1] != trajectory.shape[1]:
                raise ValueError(f"拼接失败！本地老数据的时序长度为 {old_trajectory.shape[1]}，而你当前设定的时序长度要求为 {trajectory.shape[1]}。请保持参数一致！")
            # 在 Batch 维度 (dim=0) 进行无限追加拼接
            final_trajectory = torch.cat([old_trajectory, trajectory], dim=0)
            print(f"增量拼接成功！数据池规模由 {old_trajectory.shape[0]} 条扩大至 {final_trajectory.shape[0]} 条。")
        else:
            print(f"未检测到历史文件，正在创建全新的离线数据池...")
            final_trajectory = trajectory

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'trajectory': final_trajectory}, save_path)
            print(f"数据池已安全写入本地: {save_path}")
