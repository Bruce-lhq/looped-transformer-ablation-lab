import math

import torch
import torch.nn.functional as F


LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


class Nora(torch.optim.Optimizer):
    """Normalized Orthogonal Row Alignment optimizer for scalable matrix training."""

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