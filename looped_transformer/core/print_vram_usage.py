"""显存监控工具。

提供跨平台的显存/内存占用打印函数，自动适配 MPS (Apple Silicon)、
CUDA (NVIDIA) 和 CPU 三种后端。
"""

import torch


def print_vram_usage(tag="", peak=None):
    """打印当前设备的显存/内存占用情况。

    自动适配 MPS、CUDA 和 CPU：
    - MPS：打印 PyTorch 分配内存与 Metal 驱动分配内存；
    - CUDA：打印已分配、峰值、缓存池与剩余可用；
    - CPU：提示占用主板内存。

    Args:
        tag (str): 日志标签，用于区分不同阶段的显存打印。
        peak (tuple or None): 传入 (torch_peak, metal_peak) 时显示峰值
            而非当前值（主要用于 MPS）。
    """
    if torch.backends.mps.is_available():
        # MPS 专用 API (Apple Silicon)
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver_alloc = torch.mps.driver_allocated_memory() / (1024 ** 3)
        if peak is not None:
            print(f"[{tag}] 显存占用 -> PyTorch分配: {peak[0]:.2f} GB | Metal驱动分配: {peak[1]:.2f} GB")
        else:
            print(f"[{tag}] 显存占用 -> PyTorch分配: {allocated:.2f} GB | Metal驱动分配: {driver_alloc:.2f} GB")

    elif torch.cuda.is_available():
        # CUDA 专用 API
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total - reserved
        print(f"[{tag}] 显存占用 -> 已分配: {allocated:.2f} GB | 峰值: {peak:.2f} GB | 缓存池: {reserved:.2f} GB | 剩余可用: {free:.2f} GB")

    else:
        print(f"[{tag}] 当前运行在 CPU，占用主板内存。")
