"""
Looped Transformer — 实时训练监控面板 · 后端服务
==================================================

架构概览：
  FastAPI 提供 REST + WebSocket 双通道通信。
  训练在独立线程中运行，通过 StdoutHijacker 劫持 sys.stdout 捕获日志，
  正则提取 Epoch/Loss 后推入线程安全队列，由异步桥接任务转发至 WebSocket。

线程模型：
  - Main Thread (asyncio event loop)：FastAPI 主事件循环
  - Training Thread：ExperimentTable.run() 执行的线程（通过 asyncio.to_thread 调度）
  - Queue Bridge：asyncio 任务从 queue.Queue 中轮询数据，写入 WebSocket

中断机制：
  用户点击停止 → stop_event.set() → 下一次 print() 触发 StdoutHijacker.write()
  时检测到 stop_event，通过 ctypes.PyThreadState_SetAsyncExc 向训练线程注入
  StopTraining 异常，打断 for 循环。异常被捕获后执行 offload_to_cpu() 清理显存。
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import queue
import re
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# ——— 将项目 src 目录加入 Python Path，确保能 import looped_transformer ———
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT.parent / "src" if (PROJECT_ROOT.parent / "src").exists() else PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from looped_transformer import ExperimentTable

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


# ============================================================================
# 自定义异常：用于安全中断训练线程
# ============================================================================

class StopTraining(Exception):
    """在训练线程中注入此异常以打断 for 循环。"""
    pass


# ============================================================================
# StdoutHijacker — 零侵入日志劫持器
# ============================================================================

class StdoutHijacker:
    """上下文管理器：替换 sys.stdout，捕获所有 print 输出。

    设计要点：
    1. write() 中实时正则匹配 Epoch / Loss 行，推入线程安全队列。
    2. write() 中检查 stop_event，若被设置则向当前线程注入 StopTraining 异常。
    3. 所有原始输出同时写入真实的 sys.stdout（终端可见）和内部 buffer（日志回放用）。
    4. 线程安全：queue.Queue 原生线程安全，asyncio 侧通过 run_in_executor 桥接。

    ctypes 注入原理：
      PyThreadState_SetAsyncExc(thread_id, exception) 会向指定线程的
      解释器状态中设置一个异步异常。该线程下一次执行 Python 字节码时
      （如 for 循环的下一次迭代），异常会被触发。
    """

    EPOCH_PATTERN = re.compile(
        r"Epoch\s+(\d+)/(\d+),\s+Loss:\s+([\d.eE+\-]+)"
    )
    # 匹配 "Experiment X 训练前"（来自 ExperimentTable.run 的 print_vram_usage）
    EXPERIMENT_START_PATTERN = re.compile(
        r"Experiment\s+(\d+)\s+训练前"
    )
    # 匹配 "Experiment X/Y completed"
    EXPERIMENT_DONE_PATTERN = re.compile(
        r"Experiment\s+(\d+)/(\d+)\s+completed"
    )

    def __init__(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        self.data_queue = data_queue
        self.stop_event = stop_event
        self._original_stdout: Optional[io.TextIOBase] = None
        self._buffer = io.StringIO()
        self._current_experiment = 0
        self._total_experiments = 1
        self._experiment_started = False

    def write(self, text: str):
        # 1) 原样输出到真实 stdout，方便终端调试
        self._original_stdout.write(text)
        # 2) 写入内部 buffer，用于结束后回放完整日志
        self._buffer.write(text)

        # 3) 正则提取 Epoch 信息
        match = self.EPOCH_PATTERN.search(text)
        if match:
            self.data_queue.put({
                "type": "epoch",
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "loss": float(match.group(3)),
                "experiment": self._current_experiment,
                "total_experiments": self._total_experiments,
            })

        # 4) 检测实验开始 — 提取序号并更新 _current_experiment（0-based）
        exp_start = self.EXPERIMENT_START_PATTERN.search(text)
        if exp_start:
            exp_num = int(exp_start.group(1))  # 1-based from the log
            self._current_experiment = exp_num - 1  # convert to 0-based
            self._experiment_started = True
            self.data_queue.put({
                "type": "experiment_start",
                "experiment": self._current_experiment,
                "text": text.strip(),
            })

        # 5) 检测实验完成
        exp_done = self.EXPERIMENT_DONE_PATTERN.search(text)
        if exp_done:
            self.data_queue.put({
                "type": "experiment_done",
                "experiment": int(exp_done.group(1)) - 1,
                "total": int(exp_done.group(2)),
            })

        # 6) 所有非结构化日志也推送到前端
        stripped = text.strip()
        if stripped and not match and not exp_done:
            self.data_queue.put({
                "type": "log",
                "text": stripped,
            })

        # 7) 检查停止信号 — 在训练线程中注入异常
        if self.stop_event.is_set():
            thread_id = threading.get_ident()
            result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id),
                ctypes.py_object(StopTraining),
            )
            if result == 0:
                # 线程 ID 无效
                pass
            elif result > 1:
                # 异常已被设置多次，需要恢复
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread_id), None
                )

    def flush(self):
        self._original_stdout.flush()

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._buffer = io.StringIO()
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

    def get_captured_text(self) -> str:
        """返回所有被捕获的文本。"""
        return self._buffer.getvalue()


# ============================================================================
# 全局状态管理
# ============================================================================

# 线程安全队列：训练线程写入，asyncio 桥接任务读出
_data_queue: queue.Queue = queue.Queue()

# 停止信号
_stop_event: threading.Event = threading.Event()

# 当前 ExperimentTable 引用（训练完成后用于提取结果、画图）
_current_table: Optional[ExperimentTable] = None

# 当前 result_lists 配置（供 generate_result_plots 使用）
_current_result_lists: list = [(["loss_history"], "epoch")]


# 服务状态
_server_state = {
    "status": "idle",  # idle | running | stopping | completed | error
    "message": "",
    "total_experiments": 0,
    "current_experiment": 0,
}


# ============================================================================
# 生命周期管理
# ============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期：启动时创建队列桥接任务，关闭时取消。"""
    task = asyncio.create_task(queue_bridge())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ============================================================================
# FastAPI 应用初始化
# ============================================================================

app = FastAPI(
    title="Looped Transformer Monitor",
    version="1.0.0",
    lifespan=lifespan,
)

# 静态文件服务 — 将项目根目录挂载为静态目录（index.html 放在项目根目录）
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT)), name="static")


@app.get("/")
async def root():
    """返回前端面板页面。"""
    index_path = PROJECT_ROOT / "index.html"
    return FileResponse(index_path)


# ============================================================================
# Pydantic 模型
# ============================================================================

class StartRequest(BaseModel):
    params_groups: list[dict]
    manual: Optional[dict] = None
    epochs_override: Optional[int] = None
    result_lists: Optional[list] = None  # 如 [[["loss_history"], "epoch"], [["final_loss"], "experiment"]]
    compare: Optional[bool] = True       # 横向对比模式 vs 独立双Y轴模式


class BaselinePreset(BaseModel):
    """基线实验预设定义。"""
    name: str
    params_groups: list[dict]
    manual: dict


# ============================================================================
# 基线实验预设
# ============================================================================

BASELINE_PRESETS: dict[str, BaselinePreset] = {
    "pe_comparison": BaselinePreset(
        name="位置编码对比实验",
        params_groups=[
            {"pe_type": ["learned_ape"], "experiment_name": "Learned APE"},
            {"pe_type": ["alibi"], "experiment_name": "ALiBi"},
            {"pe_type": ["rope"], "experiment_name": "RoPE"},
        ],
        manual={"epochs": 20, "print_every": 1},
    ),
    "norm_comparison": BaselinePreset(
        name="归一化方式对比",
        params_groups=[
            {"norm_type": "layernorm", "experiment_name": "LayerNorm"},
            {"norm_type": "rmsnorm", "experiment_name": "RMSNorm"},
        ],
        manual={"epochs": 20, "print_every": 1},
    ),
    "ffn_comparison": BaselinePreset(
        name="FFN 激活函数对比",
        params_groups=[
            {"ffn_type": "gelu", "experiment_name": "GELU"},
            {"ffn_type": "swiglu", "experiment_name": "SwiGLU"},
        ],
        manual={"epochs": 20, "print_every": 1},
    ),
    "gate_type_comparison": BaselinePreset(
        name="门控类型对比",
        params_groups=[
            {"residual_gate_type": "fixed", "experiment_name": "Fixed Gate"},
            {"residual_gate_type": "learnable_scalar", "experiment_name": "Learnable Scalar"},
            {"residual_gate_type": "learnable_vector", "experiment_name": "Learnable Vector"},
        ],
        manual={"epochs": 20, "print_every": 1},
    ),
    "optimizer_comparison": BaselinePreset(
        name="优化器对比",
        params_groups=[
            {"optimizer_type": "adam", "experiment_name": "Adam"},
            {"optimizer_type": "sgd", "experiment_name": "SGD"},
            {"optimizer_type": "adamw", "experiment_name": "AdamW"},
        ],
        manual={"epochs": 20, "print_every": 1},
    ),
}


# ============================================================================
# 桥接任务：从线程安全队列读取数据并转发到所有连接的 WebSocket
# ============================================================================

_ws_clients: set[WebSocket] = set()


async def queue_bridge():
    """异步桥接任务：从线程安全队列中轮询数据，广播给所有 WebSocket 客户端。

    使用 asyncio.get_event_loop().run_in_executor 将阻塞的 queue.get()
    委托给线程池执行，避免阻塞事件循环。
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            # 使用线程池执行阻塞的 queue.get，超时 0.2s 后重试
            data = await loop.run_in_executor(None, lambda: _data_queue.get(timeout=0.2))
            # 广播给所有连接的客户端
            disconnected = set()
            for ws in _ws_clients:
                try:
                    await ws.send_json(data)
                except Exception:
                    disconnected.add(ws)
            _ws_clients.difference_update(disconnected)
        except queue.Empty:
            # 超时无数据，检查是否有客户端连接
            if not _ws_clients:
                await asyncio.sleep(0.5)
            continue
        except asyncio.CancelledError:
            break


# ============================================================================
# 训练执行器
# ============================================================================

def _execute_training(
    params_groups: list[dict],
    manual: dict,
    data_queue: queue.Queue,
    stop_event: threading.Event,
    result_lists: list = None,
):
    """在独立线程中执行训练的主函数。

    设计要点：
    - 通过 manual 参数强制 print_every=1，确保每个 epoch 都有日志输出，
      从而让前端能以 epoch 级粒度实时更新 Loss 曲线，同时保证中断响应延迟 ≤ 1 epoch。
    - 使用 StdoutHijacker 上下文管理器劫持 sys.stdout。
    - 捕获 StopTraining 异常以处理用户中断。
    - 无论成功/失败/中断，都将最终状态推入队列，同时更新全局 _server_state。
    - result_lists 从前端传入，支持灵活配置绘图指标。
    """
    global _current_table, _server_state

    # 默认收集 loss_history
    if result_lists is None:
        result_lists = [(["loss_history"], "epoch")]

    # 强制 per-epoch 日志输出（用户可在 params_groups 中覆写）
    effective_manual = dict(manual)
    effective_manual.setdefault("print_every", 1)
    effective_manual.setdefault("timing", True)

    hijacker = StdoutHijacker(data_queue, stop_event)

    try:
        with hijacker:
            data_queue.put({
                "type": "status",
                "status": "initializing",
                "message": f"正在初始化 {len(params_groups)} 个实验...",
            })

            # 构建 ExperimentTable — 固定单线程模式
            table = ExperimentTable(
                params_groups=params_groups,
                manual=effective_manual,
            )
            _current_table = table

            hijacker._total_experiments = table.num_experiments

            data_queue.put({
                "type": "status",
                "status": "running",
                "message": f"开始训练 {table.num_experiments} 个实验...",
                "total_experiments": table.num_experiments,
            })

            # 执行训练 — 使用前端传入的 result_lists
            table.run(
                result_lists=result_lists,
                modes=["train"],
                parallel_workers=1,
            )

            # 正常完成
            _server_state["status"] = "completed"
            _server_state["message"] = "所有实验训练完成！"
            data_queue.put({
                "type": "status",
                "status": "completed",
                "message": "所有实验训练完成！",
            })

    except StopTraining:
        _server_state["status"] = "stopped"
        _server_state["message"] = "训练已被用户中断。"
        # 用户主动中断 — 尝试清理
        data_queue.put({
            "type": "status",
            "status": "stopped",
            "message": "训练已被用户中断。",
        })
        # 确保实验已卸载到 CPU
        if _current_table is not None:
            for exp in _current_table.experiments:
                if exp is not None and not exp.is_offloaded:
                    try:
                        exp.offload_to_cpu()
                    except Exception:
                        pass

    except Exception as e:
        _server_state["status"] = "error"
        _server_state["message"] = f"训练出错: {str(e)}"
        data_queue.put({
            "type": "status",
            "status": "error",
            "message": f"训练出错: {str(e)}",
        })
        traceback.print_exc()

    finally:
        # 无论什么状态，都将捕获的完整日志推送给前端
        captured = hijacker.get_captured_text()
        data_queue.put({
            "type": "full_log",
            "text": captured,
        })


# ============================================================================
# 结果图表生成
# ============================================================================

def generate_result_plots(
    table: ExperimentTable,
    result_lists: list,
    compare: bool = True,
) -> list[str]:
    """按 README 规则生成 Matplotlib 对比图，返回 base64 图片列表。

    Args:
        table: 已完成训练的 ExperimentTable 实例。
        result_lists: 如 [[["loss_history"], "epoch"], [["final_loss"], "experiment"]]。
        compare: True → 横向对比模式（一张图 = 一个指标，所有实验同框）。
                 False → 独立模式（一张图 = 一个 Tuple，N 子图，支持 Twinx 双 Y 轴）。

    Readme 规则：
      compare=True：遍历 metrics_list，每个指标生成一张独立的 Figure。
      compare=False：每个 Tuple 一张 Figure，划分为 num_exp 个子图。
        子图 i 中：metrics_list[0] 绑定左 Y 轴；
        若 len(metrics_list)>1，创建 ax.twinx() 绑定右 Y 轴画其余指标。
    """
    images_base64 = []

    plt.style.use("dark_background")
    bg_color = "#0a0a0f"
    text_color = "#e0e0e0"
    accent_colors = ["#00f0ff", "#b400ff", "#00ff88", "#ff6b6b",
                     "#ffd93d", "#6c5ce7", "#a29bfe", "#fd79a8"]

    if table is None or table.results_groups["train"][0] is None:
        return images_base64

    results = table.results_groups["train"]
    num_exp = table.num_experiments

    # 辅助：安全展平 metric 数据
    def _flatten(data):
        if data is None or len(data) == 0:
            return []
        if isinstance(data[0], (np.ndarray, list)):
            return [float(np.mean(d)) if hasattr(d, 'mean')
                    else float(d[0] if hasattr(d, '__iter__') and not isinstance(d, str) else d)
                    for d in data]
        return [float(d) for d in data]

    # ================================================================
    # 模式 A: compare == True — 每个指标一张独立的 Figure
    # ================================================================
    if compare:
        for _tup_idx, (metrics_list, x_type) in enumerate(result_lists):
            for metric in metrics_list:
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor(bg_color)
                ax.set_facecolor("#0d0d1a")

                if x_type == "epoch":
                    for i in range(num_exp):
                        name = table.init_parameters[i].get("experiment_name", f"Exp {i+1}")
                        flat = _flatten(results[i].get(metric, []))
                        if flat:
                            ax.plot(flat, color=accent_colors[i % len(accent_colors)],
                                    linewidth=1.8, label=name, alpha=0.9)
                    ax.set_xlabel("Epoch", color=text_color, fontsize=11)
                elif x_type == "experiment":
                    values = []
                    for i in range(num_exp):
                        v = results[i].get(metric, 0)
                        values.append(float(v) if not isinstance(v, (list, np.ndarray))
                                      else float(np.mean(v)) if v is not None else 0)
                    exp_names = [table.init_parameters[i].get("experiment_name", f"Exp {i+1}")
                                 for i in range(num_exp)]
                    bars = ax.bar(range(num_exp), values,
                                  color=[accent_colors[i % len(accent_colors)] for i in range(num_exp)],
                                  edgecolor=bg_color, linewidth=1.5, alpha=0.85)
                    ax.set_xticks(range(num_exp))
                    ax.set_xticklabels(exp_names, color=text_color, fontsize=9,
                                       rotation=30, ha="right")
                    for bar, val in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{val:.4f}", ha="center", va="bottom",
                                color=text_color, fontsize=8)

                ax.set_ylabel(metric, color=text_color, fontsize=10)
                ax.set_title(f"{metric}  (x={x_type})", color=accent_colors[0],
                             fontsize=12, fontweight="bold")
                if len(ax.get_legend_handles_labels()[0]) > 0:
                    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#333",
                              labelcolor=text_color, fontsize=8)
                ax.grid(True, alpha=0.15, color="#00f0ff")
                ax.tick_params(colors=text_color, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#333")

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, facecolor=bg_color, bbox_inches="tight")
                buf.seek(0)
                images_base64.append(base64.b64encode(buf.read()).decode())
                plt.close(fig)

    # ================================================================
    # 模式 B: compare == False — 每个 Tuple 一张 Figure，子图 + Twinx
    # ================================================================
    else:
        for _tup_idx, (metrics_list, x_type) in enumerate(result_lists):
            cols = min(num_exp, 3)
            rows = (num_exp + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            fig.patch.set_facecolor(bg_color)
            axes_flat = np.atleast_1d(axes).flatten()

            for i in range(num_exp):
                ax1 = axes_flat[i]
                ax1.set_facecolor("#0d0d1a")
                name = table.init_parameters[i].get("experiment_name", f"Exp {i+1}")

                # 左 Y 轴：metrics_list[0]
                m0 = metrics_list[0]
                flat0 = _flatten(results[i].get(m0, []))
                if flat0:
                    color0 = accent_colors[i % len(accent_colors)]
                    ax1.plot(flat0, color=color0, linewidth=1.8, label=m0, alpha=0.9)
                ax1.set_xlabel("Epoch" if x_type == "epoch" else "Experiment",
                               color=text_color, fontsize=9)
                ax1.set_ylabel(m0, color=accent_colors[0], fontsize=9)
                ax1.tick_params(axis='y', labelcolor=accent_colors[0], labelsize=7)
                ax1.tick_params(axis='x', colors=text_color, labelsize=7)
                ax1.set_title(name, color=accent_colors[0], fontsize=10, fontweight="bold")
                ax1.grid(True, alpha=0.12, color="#00f0ff")
                for spine in ax1.spines.values():
                    spine.set_color("#333")
                lines_ax1, labels_ax1 = ax1.get_legend_handles_labels()

                # 右 Y 轴（Twinx）：metrics_list[1:] 中的其余指标
                if len(metrics_list) > 1:
                    ax2 = ax1.twinx()
                    for j, mj in enumerate(metrics_list[1:], start=1):
                        flat_j = _flatten(results[i].get(mj, []))
                        if flat_j:
                            c = accent_colors[(j + 1) % len(accent_colors)]
                            ax2.plot(flat_j, color=c, linewidth=1.5,
                                     linestyle="--", label=mj, alpha=0.85)
                    ax2.set_ylabel(" & ".join(metrics_list[1:]),
                                   color=accent_colors[2], fontsize=8)
                    ax2.tick_params(axis='y', labelcolor=accent_colors[2], labelsize=7)
                    for spine in ax2.spines.values():
                        spine.set_color("#333")
                    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines_ax1 + lines_ax2, labels_ax1 + labels_ax2,
                               loc="upper right", facecolor="#1a1a2e",
                               edgecolor="#333", labelcolor=text_color, fontsize=6)
                else:
                    ax1.legend(loc="upper right", facecolor="#1a1a2e",
                               edgecolor="#333", labelcolor=text_color, fontsize=7)

            for j in range(num_exp, len(axes_flat)):
                axes_flat[j].set_visible(False)

            title_metrics = ", ".join(metrics_list)
            fig.suptitle(f"[{x_type}]  {title_metrics}", color=text_color,
                         fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, facecolor=bg_color, bbox_inches="tight")
            buf.seek(0)
            images_base64.append(base64.b64encode(buf.read()).decode())
            plt.close(fig)

    plt.style.use("default")
    return images_base64


# ============================================================================
# REST API 端点
# ============================================================================

@app.post("/api/start")
async def start_training(req: StartRequest):
    """启动训练任务。

    Request Body:
        params_groups: list[dict]  — 每个实验的参数覆写
        manual: dict | None        — 全局参数覆写
        epochs_override: int | None — 覆写所有实验的 epochs 数

    Returns:
        {"status": "started", "total_experiments": N}
    """
    global _server_state, _stop_event, _current_table

    if _server_state["status"] == "running":
        return JSONResponse(
            status_code=409,
            content={"detail": "已有训练任务正在运行"},
        )

    # 重置状态
    _stop_event.clear()
    _current_table = None

    # 清空旧队列数据
    while not _data_queue.empty():
        try:
            _data_queue.get_nowait()
        except queue.Empty:
            break

    manual = req.manual or {}
    if req.epochs_override is not None:
        manual["epochs"] = req.epochs_override

    # 原地更新（不可重新绑定新字典，否则 from server import _server_state 的引用会断裂）
    _server_state.clear()
    _server_state.update({
        "status": "running",
        "message": "训练启动中...",
        "total_experiments": len(req.params_groups),
        "current_experiment": 0,
    })

    # 构建 result_lists（若前端未传则使用默认值）
    result_lists = req.result_lists if req.result_lists else [(["loss_history"], "epoch")]
    global _current_result_lists
    _current_result_lists.clear()
    _current_result_lists.extend(result_lists)
    _server_state["compare"] = req.compare if req.compare is not None else True

    # 通过 asyncio.to_thread 将训练放入线程池执行，
    # 确保不阻塞 FastAPI 主事件循环和 WebSocket 推送。
    asyncio.create_task(
        asyncio.to_thread(
            _execute_training,
            req.params_groups,
            manual,
            _data_queue,
            _stop_event,
            result_lists,
        )
    )

    return {"status": "started", "total_experiments": len(req.params_groups)}


@app.post("/api/stop")
async def stop_training():
    """发送停止信号，中断当前训练。

    实际中断发生在下一个 print() 调用时（即下一个 epoch 结束时）。
    """
    global _server_state

    if _server_state["status"] != "running":
        return JSONResponse(
            status_code=409,
            content={"detail": "当前没有正在运行的训练任务"},
        )

    _server_state["status"] = "stopping"
    _server_state["message"] = "正在发送停止信号..."
    _stop_event.set()

    return {"status": "stopping", "message": "停止信号已发送，将在当前 epoch 结束后终止"}


@app.get("/api/state")
async def get_state():
    """获取当前服务状态。"""
    return _server_state


@app.get("/api/results")
async def get_results():
    """获取训练结果（图表 Base64 和原始数据）。

    仅在训练完成或中断后可用。
    """
    global _current_table

    if _current_table is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "没有可用的训练结果"},
        )

    try:
        images = generate_result_plots(
            _current_table,
            _current_result_lists,
            _server_state.get("compare", True),
        )

        # 提取摘要数据 + 完整原始数据（供前端交互式平滑）
        summary = []
        raw_data = {}  # {exp_index: {metric_name: [values...], ...}}
        for i in range(_current_table.num_experiments):
            expr = _current_table.results_groups["train"][i]
            name = _current_table.init_parameters[i].get("experiment_name", f"Exp {i+1}")
            summary.append({
                "index": i,
                "name": name,
                "final_loss": expr.get("loss_history", [None])[-1] if expr else None,
                "num_epochs": len(expr.get("loss_history", [])) if expr else 0,
            })
            # 收集所有可用指标的完整数值
            exp_data = {}
            if expr:
                for k, v in expr.items():
                    if v is None:
                        continue
                    # 将 numpy/tensor 转为纯 Python 列表
                    if isinstance(v, (list, np.ndarray)):
                        exp_data[k] = [
                            float(x) if hasattr(x, '__float__') or isinstance(x, (int, float, np.number))
                            else float(np.mean(x)) if hasattr(x, '__iter__') and not isinstance(x, str)
                            else x
                            for x in v
                        ]
                    else:
                        exp_data[k] = float(v) if isinstance(v, (int, float, np.number)) else v
            raw_data[str(i)] = exp_data

        return {
            "images": images,
            "summary": summary,
            "raw_data": raw_data,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"生成结果时出错: {str(e)}"},
        )


@app.get("/api/baselines")
async def get_baselines():
    """返回可用的基线实验预设列表。"""
    return {
        name: {"name": preset.name, "params_groups": preset.params_groups, "manual": preset.manual}
        for name, preset in BASELINE_PRESETS.items()
    }


@app.get("/api/parameters")
async def get_parameters():
    """返回所有可用参数的枚举值，供前端生成下拉框。"""
    # 这些枚举值来自 parameters.py 中的注释和代码逻辑
    return {
        "init": {
            "pe_type": {
                "type": "list[str]",
                "options": [
                    {"value": ["learned_ape"], "label": "Learned APE"},
                    {"value": ["ape"], "label": "APE (Fixed)"},
                    {"value": ["rope"], "label": "RoPE"},
                    {"value": ["ms_upe"], "label": "MS-UPE"},
                    {"value": ["alibi"], "label": "ALiBi"},
                ],
                "default": ["learned_ape"],
            },
            "norm_type": {
                "type": "str",
                "options": [
                    {"value": "layernorm", "label": "LayerNorm"},
                    {"value": "rmsnorm", "label": "RMSNorm"},
                ],
                "default": "layernorm",
            },
            "ffn_type": {
                "type": "str",
                "options": [
                    {"value": "gelu", "label": "GELU"},
                    {"value": "swiglu", "label": "SwiGLU"},
                ],
                "default": "gelu",
            },
            "residual_gate_type": {
                "type": "str",
                "options": [
                    {"value": "fixed", "label": "Fixed"},
                    {"value": "learnable_scalar", "label": "Learnable Scalar"},
                    {"value": "learnable_vector", "label": "Learnable Vector"},
                ],
                "default": "fixed",
            },
            "optimizer_type": {
                "type": "str",
                "options": [
                    {"value": "adam", "label": "Adam"},
                    {"value": "sgd", "label": "SGD"},
                    {"value": "adamw", "label": "AdamW"},
                ],
                "default": "adam",
            },
            "loss_type": {
                "type": "str",
                "options": [
                    {"value": "mse", "label": "MSE"},
                    {"value": "l1", "label": "L1"},
                ],
                "default": "mse",
            },
            "loop": {
                "type": "bool",
                "options": [
                    {"value": True, "label": "True (权重共享)"},
                    {"value": False, "label": "False (独立权重)"},
                ],
                "default": True,
            },
            "scheduler_type": {
                "type": "str|null",
                "options": [
                    {"value": None, "label": "None (无调度器)"},
                    {"value": "cosine", "label": "Cosine Annealing"},
                ],
                "default": None,
            },
            "data_type": {
                "type": "str",
                "options": [
                    {"value": "linear", "label": "Linear"},
                ],
                "default": "linear",
            },
        },
        "common": {
            "num_blocks": {"type": "int", "default": 20, "min": 1, "max": 100},
            "num_heads": {"type": "int", "default": 8, "min": 1, "max": 64},
            "d_model": {"type": "int", "default": 256, "min": 32, "max": 2048},
            "lr": {"type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-1},
            "epochs": {"type": "int", "default": 20, "min": 1, "max": 1000},
            "batch_size": {"type": "int", "default": 64, "min": 1, "max": 1024},
            "seq_len": {"type": "int", "default": 80, "min": 10, "max": 4096},
            "num_eff": {"type": "int", "default": 15, "min": 1, "max": 100},
            "seed": {"type": "int|null", "default": 42},
            "sink_padding": {"type": "int|null", "default": None},
            "gate_lr_ratio": {"type": "float", "default": 100, "min": 1, "max": 10000},
            "b_rope_or_upe": {"type": "float", "default": 10000, "min": 100, "max": 1000000},
            "head_ratio_upe": {"type": "float", "default": 2, "min": 1, "max": 16},
            "d_x": {"type": "int", "default": 20, "min": 1, "max": 256},
            "d_y": {"type": "int", "default": 1, "min": 1, "max": 256},
            "scheduled_training": {"type": "bool", "default": True},
        },
    }


# ============================================================================
# WebSocket 端点
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 连接端点。

    客户端连接后注册到广播列表，接收实时训练数据流。
    支持心跳检测：每 15 秒发送 ping，客户端应回复 pong。

    数据流格式（JSON）：
      {"type": "epoch", "epoch": int, "total_epochs": int, "loss": float, "experiment": int}
      {"type": "log", "text": str}
      {"type": "status", "status": str, "message": str}
      {"type": "full_log", "text": str}
    """
    await ws.accept()
    _ws_clients.add(ws)

    # 发送当前状态
    try:
        await ws.send_json({"type": "state", "state": _server_state})

        # 持续监听客户端消息（用于心跳）
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if data == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # 发送心跳
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
