"""默认配置参数模块。

提供 default_setup() 函数，集中管理所有可配置的模型初始化参数和训练参数。
"""


def default_setup(manual=None):
    """返回默认的初始化参数和训练参数字典。

    这是所有实验的"出厂默认设置"，ExperimentTable 会基于此进行局部覆写。

    Args:
        manual (dict or None): 全局参数覆写字典。key 必须在 init_parameters
            或 train_parameters 中存在，否则打印 Warning。

    Returns:
        tuple: (init_parameters, train_parameters) 两个字典。
    """
    init_parameters = {
    # RoPE/MS-UPE
        'b_rope_or_upe': 10000,            # float: RoPE或UPE的基数
        'head_ratio_upe': 2,               # float: UPE头比例

    # MultiHeadAttention
        'num_heads': 8,                    # int: 注意力头数 H
        'd_model': 256,                    # int: Transformer 的维度 D
        'max_seq_len': 100,                # int: 模型支持的最大序列长度（位置编码相关）
        'pe_type': ['learned_ape'],        # list: 位置编码类型 ('ape', 'learned_ape', 'rope', 'ms_upe', 'alibi')

    # TransformerBlock
        'norm_type': 'layernorm',          # str: transformer_block 中的归一化类型 ('layernorm' 或 'rmsnorm')
        'ffn_type': 'gelu',                # str: transformer_block 中前馈网络激活函数类型 ('gelu' 或 'swiglu')

    # ToyModel
        'num_blocks': 20,                  # int: Transformer 的层数 b
        'loop': True,                      # bool: 是否权重共享
        'residual_gate': (1, 1),           # tuple or str: transformer_block 之间传递残差的门控参数初始值
        'residual_gate_type': 'fixed',     # str: 残差门控类型 ('fixed', 'learnable_scalar', 'learnable_vector')
        'residual_random': (1, 0.1),       # tuple (mean, std): 当 residual_gate='random' 时的高斯分布参数

    # RegressionHead
        'd_x': 20,                         # int: 数据输入的特征维度
        'd_y': 1,                          # int: 数据输出的特征维度
        'bias': False,                     # bool: Regression Head 是否使用偏置
        'init_scale': None,                # float or None: Regression Head 的初始化缩放因子

    # PredictionLoss
        'loss_type': 'mse',                # str: 损失函数类型 ('mse' 或 'l1')

    # LoopedTransformerExperiment
        'lr': 1e-4,                        # float: 学习率
        'gate_lr_ratio': 100,              # float: 门控参数学习率倍数
        'seed': 42,                        # int or None: 随机种子
        'experiment_name': 'Experiment',   # str or None: 实验名称
        'timing': True,                    # bool: 是否打印初始化时间
        'optimizer_type': 'adam',          # str: 优化器类型 ('adam' 或 'sgd' 或 'adamw')
        'wd_adamw': 0.01,                  # float: AdamW优化器的权重衰减系数，仅当optimizer_type='adamw'时有效
        'task': 'regression',              # str: 任务类型 ('regression')
    }

    train_parameters = {
    # MultiHeadAttention
        'batch_size': 64,                  # int: 数据批次大小
        'seq_len': 80,                     # int: 数据序列长度

    # ToyModel
        'num_eff': 15,                     # int: 有效层数 T=b-b_0

    # dataloader
        'sink_padding': None,              # int or None: sink token 组数

    # LoopedTransformerExperiment
        'epochs': 20,                      # int: 训练轮数
        'data_type': 'linear',             # str: 回归数据类型 ('linear')
        'scheduler_type': None,            # str or None: 学习率调节器类型 ('cosine' 或 'step' 或 None)
        'lr_scale': 0.1,                   # float: 学习率调节器的缩放因子，仅当scheduler_type不为None时有效
        'step_size_scheduler': 10,         # int: StepLR调度器的步长，仅当scheduler_type='step'时有效
        'scheduled_training': True,        # bool: 是否渐进增加参与层数
        'print_every': 5,                  # int or None: 打印频率/epoch
        'timing': True,                    # bool: 是否打印训练时间
    }
    if manual is not None:
        for key, value in manual.items():
            if key in init_parameters:
                init_parameters[key] = value
            elif key in train_parameters:
                train_parameters[key] = value
            else:
                print(f"Warning: key {key} does not exist in init_parameters or train_parameters.")
    return init_parameters, train_parameters
