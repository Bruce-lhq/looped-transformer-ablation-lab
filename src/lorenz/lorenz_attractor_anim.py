import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def lorenz_derivative(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack((dx, dy, dz), dim=1)

def rk4_step(state, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    k1 = lorenz_derivative(state, sigma, rho, beta)
    k2 = lorenz_derivative(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_derivative(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_derivative(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
init_state = torch.randn(1, 3, device=device) * 15 + torch.tensor([0.0, 0.0, 25.0], device=device)

total_steps = 1500  # 设定总步数
print("▶ 正在计算洛伦兹动力学轨迹...")

with torch.no_grad():
    trajectory = torch.empty((total_steps, 1, 3), device=device)
    current_state = init_state
    for t in range(total_steps):
        current_state = rk4_step(state=current_state, dt=0.01)
        trajectory[t] = current_state

points = trajectory.squeeze(1).cpu().numpy()
x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]
print("✅ 轨迹数据生成完毕。\n")

fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

line, = ax.plot([], [], [], color='#1f77b4', lw=1.0, alpha=0.8)
head, = ax.plot([], [], [], 'ro', ms=4)  # 红色车头

step_text = ax.text2D(0.05, 0.92, "", transform=ax.transAxes, fontsize=12, 
                      color='red', fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))

ax.set_xlim((x_data.min() - 5, x_data.max() + 5))
ax.set_ylim((y_data.min() - 5, y_data.max() + 5))
ax.set_zlim((z_data.min() - 5, z_data.max() + 5))
ax.set_title("Lorenz Attractor Real-time Simulation", fontsize=12, fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=45)

def update(num):
    # 每 10 步（或第 1 步）精准打印带三维坐标的日志
    if (num + 1) % 10 == 0 or num == 0:
        print(f"⏳ [Step {num + 1}/{total_steps}] 坐标位置 -> X: {x_data[num]:.2f}, Y: {y_data[num]:.2f}, Z: {z_data[num]:.2f}")

    # 更新线段和车头
    line.set_data(x_data[:num], y_data[:num])
    line.set_3d_properties(z_data[:num])
    head.set_data([x_data[num]], [y_data[num]])
    head.set_3d_properties([z_data[num]])
    
    # 更新画布内实时跳动的文本
    step_text.set_text(f"Step: {num + 1}")
    
    return line, head, step_text

ani = animation.FuncAnimation(fig, update, frames=total_steps, interval=5, blit=True)

print("💾 正在将动画帧写入磁盘，生成 GIF 文件（这会触发完整的渲染日志）...")
gif_path = 'lorenz_attractor.gif'
# 这里会安全地一边渲染一边写盘
ani.save(gif_path, writer='pillow', fps=60)
print(f"🎉 动图已安全落盘，文件生成成功: '{gif_path}'\n")
print("▶ 正在开启实时动画交互窗口...")
plt.show()