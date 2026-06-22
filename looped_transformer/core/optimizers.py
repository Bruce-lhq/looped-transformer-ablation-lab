"""优化器模块。

提供两个组件：
- ``HybridOptimizer``：多优化器包装器，把若干独立 optimizer 组合成一个对象，对外暴露
  统一的 ``zero_grad``/``step``/``state_dict``/``state``/``param_groups`` 接口，
  使它们在每个训练步同步前进。用于让 Muon/Nora（作用于 ≥2 维参数）与 AdamW
  （作用于其余参数）作为整体被 LoopedTransformerExperiment 调度。
- ``Nora``：Normalized Orthogonal Row Alignment 优化器，迁自原根目录 ``nora.py``。
  对 ≥2 维参数，将动量更新方向对参数的行向量做正交化（投影到与当前参数正交的方向），
  再按 sqrt(in/out) 缩放；对 1 维/标量参数回退为 Adam。支持低精度训练下的 fp32 master param。
"""

import math

import torch
import torch.nn.functional as F


LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


class HybridOptimizer:
    """多优化器包装器。

    将多个独立的 optimizer 组合为一个对象，使它们在每个训练步同步前进。
    用于把作用于不同参数组的优化器（如 Muon/Nora 作用于 ≥2 维矩阵参数，
    AdamW 作用于其余参数与门控参数）当作一个整体使用。

    实现的接口与 ``torch.optim.Optimizer`` 对齐：``zero_grad``、``step``、
    ``state_dict``、``load_state_dict``，以及 ``state`` / ``param_groups`` 属性。
    """

    def __init__(self, optimizers):
        """初始化 HybridOptimizer。

        Args:
            optimizers (list): 被包装的 optimizer 列表，按顺序组合。
        """
        self.optimizers = optimizers

    def zero_grad(self):
        """对所有被包装的 optimizer 调用 zero_grad。"""
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        """对所有被包装的 optimizer 调用 step（同步前进一个训练步）。"""
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        """收集所有被包装 optimizer 的 state_dict。

        Returns:
            list: 各 optimizer 的 state_dict，顺序与构造时一致。
        """
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts):
        """逐个恢复被包装 optimizer 的状态。

        Args:
            state_dicts (list): 各 optimizer 的 state_dict，顺序与构造时一致。
        """
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)

    @property
    def state(self):
        """合并所有被包装 optimizer 的 state 字典。"""
        combined = {}
        for optimizer in self.optimizers:
            combined.update(optimizer.state)
        return combined

    @property
    def param_groups(self):
        """拼接所有被包装 optimizer 的 param_groups。"""
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups


class Nora(torch.optim.Optimizer):
    """Normalized Orthogonal Row Alignment optimizer for scalable matrix training.

    对每个参数张量，根据 ``is_nora`` 标志和参数维度选择更新规则：

    - **NORA 分支**（``is_nora=True`` 且 ``grad.dim() >= 2``）：
      维护动量缓冲 ``buf = lerp(buf, grad, 1-beta)`` 与合成方向 ``m_t = lerp(grad, buf, momentum)``；
      将当前参数行向量归一化为 ``theta_hat``，把 ``m_t`` 投影到与 ``theta_hat`` 正交的方向 ``v``，
      再归一化为 ``v_hat`` 并按 ``sqrt(in_features/out_features)`` 缩放作为更新方向。
    - **Adam 回退**（其余参数）：标准的 Adam（带 bias correction），用于 1 维/标量参数。

    对低精度参数（fp16/bf16）维护一份 fp32 master param，更新后写回原精度。
    """

    def __init__(
        self,
        param_groups,
        lr_nora=0.005,
        lr_adam=0.001,
        momentum=0.95,
        beta=0.95,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-10,
    ):
        """初始化 Nora。

        Args:
            param_groups (list): 参数组列表，每组可用 ``is_nora`` 标志指定是否走 NORA 分支。
            lr_nora (float): NORA 分支的学习率。
            lr_adam (float): Adam 回退分支的学习率。
            momentum (float): NORA 合成方向的动量系数。
            beta (float): NORA 动量缓冲的 EMA 系数。
            weight_decay (float): 权重衰减系数。
            betas (tuple): Adam 回退分支的 (beta1, beta2)。
            eps (float): 归一化与分母的数值稳定项。
        """
        defaults = dict(
            lr_nora=lr_nora,
            lr_adam=lr_adam,
            momentum=momentum,
            beta=beta,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(param_groups, defaults)

    def step(self, closure=None):
        """执行一步参数更新。

        Args:
            closure (callable or None): 可选的闭包，用于重算 loss（多数训练循环不使用）。

        Returns:
            loss or None: closure 返回的 loss（若提供）。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group.get("momentum", 0.95)
            beta = group.get("beta", 0.95)
            weight_decay = group.get("weight_decay", 0.0)
            betas = group.get("betas", (0.9, 0.95))
            eps = group.get("eps", 1e-10)
            is_nora = group.get("is_nora", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_state = self.state.setdefault(p, {})

                use_master_param = p.data.dtype in LOW_PRECISION_DTYPES
                if use_master_param:
                    if "fp32_param" not in param_state:
                        param_state["fp32_param"] = p.data.detach().float().clone()
                    elif param_state["fp32_param"].dtype != torch.float32:
                        param_state["fp32_param"] = param_state["fp32_param"].float()
                    param_data = param_state["fp32_param"]
                    grad_data = grad.float()
                else:
                    param_data = p.data
                    grad_data = grad

                if is_nora and grad.dim() >= 2:
                    if "momentum_buffer" not in param_state:
                        buf = torch.zeros_like(grad_data)
                    else:
                        buf = param_state["momentum_buffer"]
                        if use_master_param and buf.dtype != torch.float32:
                            buf = buf.float()

                    buf.lerp_(grad_data, 1 - beta)
                    m_t = grad_data.lerp(buf, momentum)

                    theta_hat = F.normalize(param_data, p=2, dim=-1, eps=eps)

                    dot_product = torch.sum(m_t * theta_hat, dim=-1, keepdim=True)
                    v = m_t - dot_product * theta_hat

                    v_hat = F.normalize(v, p=2, dim=-1, eps=eps)

                    scale = max(1, math.sqrt(grad_data.size(-2) / grad_data.size(-1)))
                    update_direction = v_hat * scale

                    if weight_decay > 0:
                        param_data.mul_(1 - lr * weight_decay)

                    param_data.add_(update_direction, alpha=-lr)

                    if use_master_param:
                        p.data.copy_(param_data.to(dtype=p.data.dtype))

                    param_state["momentum_buffer"] = buf

                else:
                    if "exp_avg" not in param_state:
                        param_state["exp_avg"] = torch.zeros_like(grad_data)
                        param_state["exp_avg_sq"] = torch.zeros_like(grad_data)
                        param_state["step"] = 0
                    elif use_master_param and param_state["exp_avg"].dtype != torch.float32:
                        param_state["exp_avg"] = param_state["exp_avg"].float()
                        param_state["exp_avg_sq"] = param_state["exp_avg_sq"].float()

                    exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
                    param_state["step"] += 1

                    exp_avg.mul_(betas[0]).add_(grad_data, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad_data, grad_data, value=1 - betas[1])

                    bias_correction1 = 1 - betas[0] ** param_state["step"]
                    bias_correction2 = 1 - betas[1] ** param_state["step"]
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom

                    if weight_decay > 0:
                        param_data.mul_(1 - step_size * weight_decay)

                    param_data.add_(adam_update, alpha=-step_size)

                    if use_master_param:
                        p.data.copy_(param_data.to(dtype=p.data.dtype))

        return loss


def get_nora_optimizer(
    model,
    lr_nora=0.005,
    lr_adam=0.001,
    weight_decay=0.1,
    momentum=0.95,
    beta=0.95,
):
    """为模型构造一个 Nora 优化器。

    参数分组：≥2 维参数（排除名字含 ``embed`` / ``lm_head`` 的）走 NORA 分支，
    其余（1 维/标量/embedding/head）走 Adam 回退分支。

    Args:
        model (nn.Module): 待优化的模型。
        lr_nora (float): NORA 分支学习率。
        lr_adam (float): Adam 回退分支学习率。
        weight_decay (float): 权重衰减系数。
        momentum (float): NORA 合成方向的动量系数。
        beta (float): NORA 动量缓冲的 EMA 系数。

    Returns:
        Nora: 配置好的优化器实例。
    """
    nora_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                nora_params.append(param)
            else:
                adam_params.append(param)

    param_groups = [
        dict(
            params=nora_params,
            lr=lr_nora,
            lr_nora=lr_nora,
            lr_adam=lr_adam,
            weight_decay=weight_decay,
            momentum=momentum,
            beta=beta,
            is_nora=True,
        ),
        dict(
            params=adam_params,
            lr=lr_adam,
            lr_nora=lr_nora,
            lr_adam=lr_adam,
            weight_decay=weight_decay,
            momentum=momentum,
            beta=beta,
            is_nora=False,
        ),
    ]
    optimizer = Nora(param_groups)
    return optimizer
