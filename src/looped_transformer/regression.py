"""下游任务模块：线性回归。

提供 RegressionHead（双通道映射 + 拉链式 Prompt 拼接）、
PredictionLoss（损失计算 + 预测读取）、
RegressionSolver（端到端拼装）。
"""

import torch
import torch.nn as nn

from .toy_model import ToyModel


class RegressionHead(nn.Module):
    """回归任务的特征映射头。

    将 x（特征）和 y（标签）分别通过独立线性层映射到 d_model 维度，
    然后"拉链式"交织拼接成 Prompt 序列：x_1, y_1, x_2, y_2, ..., x_k, y_k。

    Attributes:
        read_in_x (nn.Linear): x 的特征映射层 [d_x -> d_model]。
        read_in_y (nn.Linear): y 的特征映射层 [d_y -> d_model]。
    """

    def __init__(self, d_model, d_x, d_y=1, bias=False, init_scale=None):
        """初始化 RegressionHead。

        Args:
            d_model (int): 目标 embedding 维度。
            d_x (int): 输入特征 x 的原始维度。
            d_y (int): 输出标签 y 的原始维度，默认 1。
            bias (bool): 线性层是否使用偏置，默认 False。
            init_scale (float or None): 权重初始化标准差，None 使用默认初始化。
        """
        super().__init__()
        self.read_in_x = nn.Linear(d_x, d_model, bias=bias)
        self.read_in_y = nn.Linear(d_y, d_model, bias=bias)
        if init_scale is not None:
            nn.init.normal_(self.read_in_x.weight, mean=0.0, std=init_scale)
            nn.init.normal_(self.read_in_y.weight, mean=0.0, std=init_scale)

    def forward(self, x_data, y_data):
        """将独立的 x、y 批次映射为交织的 Prompt 序列。

        Args:
            x_data (torch.Tensor): 输入特征 [batch_size, k, d_x]。
            y_data (torch.Tensor): 输入标签 [batch_size, k, d_y]。

        Returns:
            torch.Tensor: 交织后的 Prompt [batch_size, 2k, d_model]。
        """
        x_emb = self.read_in_x(x_data)
        y_emb = self.read_in_y(y_data)
        stacked = torch.stack((x_emb, y_emb), dim=2)
        Prompt = stacked.flatten(1, 2)
        return Prompt


class PredictionLoss(nn.Module):
    """预测损失模块。

    包含 read_out 线性层（d_model -> d_y）和 MSE/L1 损失函数。
    训练模式计算所有有效层的损失（支持 sink_padding 遮蔽），
    评估模式仅返回最后一层最后位置的预测值。

    Attributes:
        read_out (nn.Linear): 输出投影层。
        loss_fn (nn.Module): MSE 或 L1 损失函数。
        weight_decay (float): 权重衰减系数（预留，当前未直接使用）。
    """

    def __init__(self, d_model, d_y=1, loss_type='mse', layer_weight_decay=1.0, seq_weight_decay=1.0):
        """初始化 PredictionLoss。

        Args:
            d_model (int): 输入维度。
            d_y (int): 输出维度，默认 1。
            loss_type (str): 损失类型，'mse' 或 'l1'。
            layer_weight_decay (float): 层衰减系数，后续层权重更大。
            seq_weight_decay (float): 序列位置衰减系数，后面位置权重更大。
        """
        super().__init__()
        self.read_out = nn.Linear(d_model, d_y)
        self.layer_weight_decay = layer_weight_decay
        self.seq_weight_decay = seq_weight_decay
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, outputs, y_true, is_eval=False, sink_padding=None):
        """计算损失或返回预测值。

        Args:
            outputs (torch.Tensor): ToyModel 输出
                [batch_size, num_eff, seq_len, d_model]。
            y_true (torch.Tensor): 真实标签 [batch_size, k, d_y]。
            is_eval (bool): True 时仅返回最后一个位置的预测值（评估模式）。
            sink_padding (int or None): sink token 组数，用于在损失计算时
                跳过前 sink_padding 个位置。

        Returns:
            训练模式返回 (loss, y_pred_norm, y_true_norm) 三元组；
                评估模式返回最终预测 [batch_size, d_y]。
        """
        if is_eval:
            y_outputs = outputs[:, -1, -2, :]
            y_pred_final = self.read_out(y_outputs)
            return y_pred_final
        y_outputs = outputs[:, :, 0::2, :]
        y_preds = self.read_out(y_outputs)
        if sink_padding is not None:
            y_preds = y_preds[:, :, sink_padding:, :]
            y_true = y_true[:, sink_padding:, :]
        y_pred_norm = torch.sqrt(torch.mean(y_preds.detach() ** 2)).item()
        y_true_norm = torch.sqrt(torch.mean(y_true.detach() ** 2)).item()
        loss_unreduced = self.loss_fn(y_preds, y_true.unsqueeze(1).expand_as(y_preds))
        if self.layer_weight_decay != 1.0:
            num_eff = outputs.shape[1]
            layer_weights = self.layer_weight_decay ** torch.arange(num_eff - 1, -1, -1, device=outputs.device)
            layer_weights = layer_weights / layer_weights.mean()
            loss_unreduced = loss_unreduced * layer_weights.view(1, num_eff, 1, 1)
        if self.seq_weight_decay != 1.0:
            k = loss_unreduced.shape[2]
            seq_weights = self.seq_weight_decay ** torch.arange(k - 1, -1, -1, device=outputs.device)
            seq_weights = seq_weights / seq_weights.mean()
            loss_unreduced = loss_unreduced * seq_weights.view(1, 1, k, 1)
        return loss_unreduced.mean(), y_pred_norm, y_true_norm


class RegressionSolver(nn.Module):
    """回归任务端到端求解器。

    将 ToyModel、RegressionHead、PredictionLoss 拼装为一个完整模型。
    前向传播：x,y → Head → Prompt → ToyModel → outputs → PredictionLoss → loss。

    Attributes:
        toy_model (ToyModel): Looped Transformer 引擎。
        head (RegressionHead): 输入映射头。
        loss_fn (PredictionLoss): 损失计算模块。
    """

    def __init__(self, num_blocks, num_heads, d_model, d_x, d_y, max_seq_len,
                 norm_type='layernorm', ffn_type='gelu', pe_type='learned_ape',
                 b_rope_or_upe=10000, head_ratio_upe=2,
                 loop=True, loss_type='mse',
                 residual_gate=(1, 1), residual_gate_type='fixed',
                 residual_random=(1, 0.1),
                 bias=False, init_scale=None,
                 layer_weight_decay=1.0, seq_weight_decay=1.0):
        """初始化 RegressionSolver。

        Args:
            num_blocks (int): Looped Transformer 总迭代层数。
            num_heads (int): 注意力头数。
            d_model (int): 模型维度。
            d_x (int): 输入特征维度。
            d_y (int): 输出标签维度。
            max_seq_len (int): 最大序列长度。
            norm_type (str): 归一化类型。
            ffn_type (str): FFN 类型。
            pe_type (str or list[str]): 位置编码类型。
            b_rope_or_upe (int): RoPE/MS_UPE 基频。
            head_ratio_upe (int): MS_UPE 头倍率。
            loop (bool): 是否权重共享。
            loss_type (str): 损失函数类型，'mse' 或 'l1'。
            residual_gate (tuple or str): 残差门控初始值。
            residual_gate_type (str): 门控类型。
            residual_random (tuple): 随机初始化的 (mean, std)。
            bias (bool): RegressionHead 是否使用偏置。
            init_scale (float or None): RegressionHead 初始化标准差。
            layer_weight_decay (float): 层衰减系数。
            seq_weight_decay (float): 序列位置衰减系数。
        """
        super().__init__()
        self.toy_model = ToyModel(
            num_blocks, num_heads, d_model, max_seq_len,
            norm_type=norm_type, ffn_type=ffn_type, pe_type=pe_type,
            loop=loop, b_rope_or_upe=b_rope_or_upe, head_ratio_upe=head_ratio_upe,
            residual_gate=residual_gate, residual_gate_type=residual_gate_type,
            residual_random=residual_random,
        )
        self.head = RegressionHead(d_model, d_x, d_y, bias=bias, init_scale=init_scale)
        self.loss_fn = PredictionLoss(d_model, d_y, loss_type=loss_type,
                                      layer_weight_decay=layer_weight_decay,
                                      seq_weight_decay=seq_weight_decay)

    def forward(self, x_data, y_data, num_eff, current_blocks=None, is_eval=False, sink_padding=None):
        """前向传播：完整的数据 → 预测 → 损失流程。

        Args:
            x_data (torch.Tensor): 输入特征 [batch_size, k, d_x]。
            y_data (torch.Tensor): 真实标签 [batch_size, k, d_y]。
            num_eff (int): 有效层数 T。
            current_blocks (int or None): 实际执行的迭代数。
            is_eval (bool): 是否评估模式。
            sink_padding (int or None): sink token 组数。

        Returns:
            评估模式: 预测值 [batch_size, d_y]。
            训练模式: (loss, y_pred_norm, y_true_norm) 三元组。
        """
        Prompt = self.head(x_data, y_data)
        outputs = self.toy_model(Prompt, num_eff, current_blocks=current_blocks)
        output = self.loss_fn(outputs, y_data, is_eval=is_eval, sink_padding=sink_padding)
        return output
