"""默认配置参数模块。

提供 ``default_setup(manual=None)``，集中管理所有可配置的模型初始化参数、训练参数，
以及评估配置（``eval_config`` 仅作文档说明 ``ood_kwargs`` 的可选键）。
"""

import torch.nn.functional as F


def default_setup(manual=None):
    """返回默认的初始化参数与训练参数字典。

    这是所有实验的"出厂默认设置"，``ExperimentTable`` 会基于此做局部覆写。

    Args:
        manual (dict or None): 全局参数覆写字典。key 必须在 init_parameters 或
            train_parameters 中存在，否则打印 Warning。

    Returns:
        tuple: (init_parameters, train_parameters) 两个字典。
    """
    init_parameters = dict(
    # RoPE/MS-UPE
        b_rope_or_upe=10000,            # float: RoPE或UPE的基数
        head_ratio_upe=2,               # float: UPE头比例

    # MultiHeadAttention
        num_heads=8,                    # int: 注意力头数 H
        d_model=256,                    # int: Transformer 的维度 D
        max_seq_len=100,                # int: 模型支持的最大序列长度（位置编码相关）
        pe_type=['learned_ape'],        # list: 位置编码类型 ('ape', 'learned_ape', 'rope', 'ms_upe', 'alibi')

    # SinkMetricsProbe
        sink_threshold=0.3,             # float: 判断阈值，sink rate 表示所有 token 在第0个位置的注意力的平均值超过这个阈值的头数比例

    # TransformerBlock
        norm_type='layernorm',          # str: transformer_block 中的归一化类型 ('layernorm' 或 'rmsnorm')
        ffn_type='gelu',                # str: transformer_block 中前馈网络激活函数类型 ('gelu' 或 'swiglu')

    # ToyModel
        num_blocks=20,                  # int: Transformer 的层数 b
        loop=True,                      # bool: 是否权重共享
        residual_gate=(1, 1),           # tuple or str: transformer_block 之间传递残差的门控参数初始值,使`x=a*x+b*x_0` ((a,b)或'random')
        residual_gate_type='fixed',     # str: 残差门控类型 ('fixed', 'learnable_scalar', 'learnable_vector')
        residual_random=(1, 0.1),       # tuple (mean, std): 当 residual_gate='random' 时满足高斯分布 N(mean, std)
        x_init='zero',                  # str: x在所有block之前的初始化方式 ('prompt'(x=x_0) 或 'zero'(x=0))

    # RegressionHead
        d_x=20,                         # int: 数据输入的特征维度
        d_y=1,                          # int: 数据输出的特征维度
        bias=False,                     # bool: Regression Head 是否使用偏置
        init_scale=None,                # float or None: Regression Head 的初始化缩放因子(None表示使用默认初始化)。仅当init_std为None时有效，如果init_std不为None，则使用init_std进行权重初始化，并忽略init_scale。

    # RegressionSolver
        init_std=0.02,                  # float or 'auto' or None: 所有nn.Linear和nn.Embedding权重初始化的标准差 ('auto'表示直接取1/sqrt(d_model)，None表示使用默认初始化)

    # PredictionLoss
        loss_type='mse',                # str: 损失函数类型 ('mse' 或 'l1')
        layer_weight_decay=0.8,         # float: 层权重衰减因子，控制不同层的损失权重 (默认0.8表示从最后一层到第一层递减的权重，1.0表示不加权)
        seq_weight_decay=0.8,           # float: 序列权重衰减因子，控制不同时间步的损失权重 (默认0.8表示从最后一次预测到第一次预测递减的权重，1.0表示不加权)

    # LoopedTransformerExperiment
        lr=1e-4,                        # float: 学习率
        lr_muon=1e-3,                   # float: Muon的学习率
        lr_nora=5e-4,                   # float: Nora的学习率
        gate_lr_ratio=100,              # float: 当使用残差门控时，门控参数的学习率相对于模型其他参数的倍数
        seed=42,                        # int or None: 手动随机种子，确保实验可复现
        experiment_name='Experiment',   # str or None: 实验名称 (None表示不打印设备和实验初始化信息)
        timing=True,                    # bool: 是否打印初始化时间
        print_on=True,                  # bool: 是否打印初始化信息
        optimizer_type='adamw',         # str: 优化器类型 ('*' 或 'muon_*' 或 'nora_*'('*' 为 'adamw' 或 'adam' 或 'sgd',默认'sgd'))
        wd_adamw=0.01,                  # float: AdamW优化器的权重衰减系数，仅当optimizer_type='adamw'时有效
        task='regression',              # str: 任务类型 ('regression')
        load_path=None,                 # str or 'auto'(指'saved_checkpoints/<safe_experiment_name>.pth') or None: 预训练模型路径 (None表示不加载预训练模型)
    )

    train_parameters = dict(
    # MultiHeadAttention
        batch_size=64,                  # int: 数据批次大小
        seq_len=80,                     # int: 数据序列长度

    # ToyModel
        num_eff=15,                     # int: 有效层数，即参与误差计算、参与梯度反向传播的层数 T=b-b_0

    # dataloader
        sink_padding=None,              # int or None: seq 最前面填充的 sink token 组数 (None表示不使用sink padding)
        d_hidden=64,                    # int: 非线性数据生成器中隐藏层的维度，仅当data_type='nonlinear'时有效
        function_callable=lambda x: 2**0.5 * F.relu(x),  # function: 非线性数据生成器中使用的非线性函数，仅当data_type='nonlinear'时有效(系数2**0.5是为了使方差期望变为1)
        lorenz_kwargs=dict(dt=0.01, burn_in=500),  # dict: 洛伦兹系统数据生成的额外参数，仅当data_type='lorenz'时有效
                                        # lorenz_kwargs的可选键包括：
                                        # 'dt': float, 洛伦兹系统的时间步长
                                        # 'burn_in': int, 洛伦兹系统的预热步数，即在正式生成训练数据之前，先运行系统burn_in步以达到吸引子区域
        load_lorenz_from=None,          # str or None: 洛伦兹系统数据生成的初始状态加载路径 (None表示随机初始化)

    # LoopedTransformerExperiment
        epochs=100,                     # int: 训练轮数
        steps_per_epoch=1,              # int: 每轮训练的步数
        data_type='linear',             # str: 回归数据类型 ('linear'或'nonlinear'或'lorenz')
                                        # 注意：当data_type='lorenz'时，务必令d_x=3,d_y=3以匹配洛伦兹系统的状态空间维度
        scheduler_type=None,            # str or None: 学习率调节器类型 ('cosine' 或 'step' 或 None)
        eta_min=1e-5,                   # float: CosineAnnealingLR调度器的最小学习率，仅当scheduler_type='cosine'时有效
        lr_scale=0.1,                   # float: 学习率调节器的缩放因子，仅当scheduler_type不为None时有效
        step_size_scheduler=10,         # int: StepLR调度器的步长，仅当scheduler_type='step'时有效
        scheduled_training=False,       # bool: 是否使用 kick-start，即随着训练的进行逐渐增加参与的层数 b=epoch+num_eff ，直到达到最大层数num_blocks（用处并不大）
        curriculum=None,                # dict or None: 课程学习配置 (None表示不使用课程学习)
                                        # curriculum的可选键包括：
                                        # 'd_x': int, curriculum learning 的初始有效d_x
                                        # 'seq_len': int, curriculum learning 的初始有效seq_len
                                        # 'duration_ratio': float, curriculum learning 的持续时间占比，取值范围为0到1，表示在训练的前duration_ratio比例的时间内逐渐增加有效d_x或seq_len，之后保持不变。
        print_every=20,                 # int or None: 打印频率/epoch (None表示不打印训练过程中的损失信息)
        timing=True,                    # bool: 是否打印训练时间
        save_path=None                  # str or 'auto'(指'saved_checkpoints/<safe_experiment_name>.pth') or None: 训练完成后模型的保存路径 (None表示不保存模型)
    )

    eval_config = dict(
        eval_name='eval',               # str: 评估名称
        ood_kwargs=None,                # dict or None: 用于OOD数据生成的额外参数，仅evaluate时有效。
                                        # ood_kwargs的可选键包括：
                                        # 'x_distribution': str, 输入特征的分布类型 ('gaussian', 'uniform', 'laplace') # x 分布偏移
                                        # 'x_scale': float, 输入特征的尺度调整因子                                      # x 尺度偏移
                                        # 'x_mean_shift': float, 输入特征的均值平移量                                   # x 均值偏移
                                        # 'x_noise_std': float, 输入特征的噪声标准差                                    # x 噪声鲁棒性
                                        # 'y_scale': float, 标签的尺度调整因子                                          # y 尺度偏移
                                        # 'y_mean_shift': float, 标签的均值平移量                                       # y 均值偏移
                                        # 'y_noise_std': float, 标签的噪声标准差                                        # y 噪声鲁棒性
                                        # 'function_callable': function, 数据生成函数，仅当data_type='nonlinear'时有效   # 函数偏移
                                        # 'seq_len_scale': float, evaluate时序列长度缩放因子                            # 序列长度外推
                                        # 'init_distribution': str,                                # (for lorenz attractor)初始状态分布类型
                                        # 'lorenz_noise_std': float, 洛伦兹系统噪声标准差             # (for lorenz attractor)噪声鲁棒性
                                        # 'sigma_shift': float, 洛伦兹系统的sigma参数偏移量           # (for lorenz attractor)参数偏移
                                        # 'rho_shift': float, 洛伦兹系统的rho参数偏移量               # (for lorenz attractor)参数偏移
                                        # 'beta_shift': float, 洛伦兹系统的beta参数偏移量             # (for lorenz attractor)参数偏移
    )

    if manual is not None:
        for key, value in manual.items():
            if key in init_parameters:
                init_parameters[key] = value
            elif key in train_parameters:
                train_parameters[key] = value
            else:
                print(f"Warning:key {key} do not exist in init_parameters and train_parameters.")
    return init_parameters, train_parameters
