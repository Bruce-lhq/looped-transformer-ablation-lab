"""多实验调度与可视化模块。

``ExperimentTable`` 是多实验对比的核心调度器：传入一组参数字典 → 自动跑完所有实验
→ 收集指标 → 生成对比图。支持多线程并行、训练/评估双模式、双 Y 轴绑图（result_lists
中用 ``'|'`` 分隔左右轴指标）、epoch/block 折线与 experiment 柱状图、baseline 相对值。
"""

import copy
import math
import threading
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import torch

from .default_setup import default_setup
from .experiment import LoopedTransformerExperiment
from .core.print_vram_usage import print_vram_usage
from .data_generators import clear_lorenz_cache


class ExperimentTable:
    """多实验调度与自动画图引擎。

    底层逻辑是"全量默认 + 局部覆写"：自动调用 ``default_setup()`` 生成默认参数，
    ``params_groups`` 中填什么就覆写什么。支持串行/并行执行、训练/评估双模式、
    自动对比绑图（折线 / 柱状 / 双 Y 轴 / baseline 相对值）。

    Attributes:
        num_experiments (int): 实验总数。
        init_parameters (list[dict]): 每个实验的初始化参数。
        train_parameters (list[dict]): 每个实验的训练参数。
        result_lists (list[tuple]): 用户指定的需要收集/画图的数据配置。
        results (list[dict] or None): 收集到的结果，每个 dict 是一个实验的 metric→value。
        experiments (list[LoopedTransformerExperiment]): 实验对象列表（已转移至 CPU）。
    """

    def __init__(self, params_groups: list[dict], manual=None):
        """初始化实验台并构造所有实验对象。

        Args:
            params_groups (list[dict]): 每个元素是一个实验的参数覆写字典。未指定的 key
                使用 ``default_setup()`` 的默认值。每个字典必须使用 init_parameters
                或 train_parameters 中存在的 key。
            manual (dict or None): 全局参数覆盖，在逐实验覆写之前应用到所有实验。

        Raises:
            ValueError: 当 params_groups 中存在未知参数名时。
        """
        init, train = default_setup(manual=manual)
        self.num_experiments = len(params_groups)
        self.init_parameters = [copy.deepcopy(init) for _ in range(self.num_experiments)]
        self.train_parameters = [copy.deepcopy(train) for _ in range(self.num_experiments)]
        # 自动分类传入的参数以更新默认参数
        for i, params in enumerate(params_groups):
            for key, value in params.items():
                if key in self.init_parameters[i]:
                    self.init_parameters[i][key] = value
                elif key in self.train_parameters[i]:
                    self.train_parameters[i][key] = value
                else:
                    raise ValueError(f"Unknown parameter: {key}")
                if key == 'print_every' and value is None:
                    self.init_parameters[i]['print_on'] = False
        self.experiments = [None] * self.num_experiments
        for i in range(self.num_experiments):
            experiment = LoopedTransformerExperiment(**self.init_parameters[i])
            experiment.offload_to_cpu()  # 初始化后将模型移回CPU，节省GPU显存
            self.experiments[i] = experiment

    def run(self, result_lists: list[tuple[list[str], str, int]], modes=['train'], parallel_workers=1, eval_configs=None):
        """执行所有实验并收集结果。

        根据 ``parallel_workers`` 决定串行或并行执行。并行模式下自动关闭日志输出以避免
        竞态。每个实验完成后自动 ``offload_to_cpu()`` 并清空 GPU 缓存。

        Args:
            result_lists (list[tuple]): 要收集/画图的数据配置。
                格式：``[(['指标1', '指标2'], '横轴类型'), ...]`` 或
                ``[(['指标1', '指标2'], '横轴类型', baseline_index), ...]``。
                列表中可用 ``'|'`` 作为双 Y 轴分隔符——其左侧指标绑左轴，右侧绑右轴。
                横轴类型：``'epoch'``（训练过程，metric 须为列表）、``'block'``（模型深度，
                仅限 eval 与 captured 的 sink 指标）、``'experiment'``（汇总柱状图，metric
                为单个值）。
            modes (list[str]): 运行模式，``'train'`` 和/或 ``'evaluate'``。
            parallel_workers (int): 并行线程数。1 为串行，>1 为多线程并行。
            eval_configs (list[dict] or None): 评估配置，形如
                ``[{'eval_name': 'id'}, {'eval_name': 'ood_x_scale', 'ood_kwargs': {'x_scale': 2.0}}]``。
                eval_name 会作为评估结果中各指标的前缀（如 id_loss、ood_x_scale_y_pred_norm）。

        Raises:
            ValueError: parallel_workers 不是正整数时。
        """
        # 创建实验对象并运行
        result_set = set(key for item in result_lists for key in item[0] if key != '|')  # 兼容长度为2或3的 tuple，获取所有需要展示的结果名称的集合
        self.result_lists = result_lists
        self.results = [None] * self.num_experiments
        if eval_configs is None:
            eval_configs = []
        print_lock = threading.Lock()  # 用于在线程中安全地打印日志
        vram_printed = False

        def single_train(i, parallel=False):
            nonlocal vram_printed
            eval_count = 0
            experiment = self.experiments[i]
            experiment.load_to_device()  # 训练前将模型加载到GPU
            if parallel:
                self.train_parameters[i]['timing'] = False
                self.train_parameters[i]['print_every'] = None
            if parallel:
                with print_lock:
                    if not vram_printed and self.init_parameters[i]['print_on']:
                        print_vram_usage(tag=f"Parallel峰值显存检测")
                        vram_printed = True
            else:
                if self.init_parameters[i]['print_on']:
                    print_vram_usage(tag=f"Experiment {i+1} 训练前")
            for mode in modes:
                if mode == 'train':
                    experiment.train(**self.train_parameters[i])
                    self.results[i] = experiment.get_results(result_set)
                    if parallel:
                        with print_lock:
                            if self.init_parameters[i]['print_on']:
                                print_vram_usage(tag=f"Experiment {i+1} 训练峰值", peak=(experiment.max_torch_vram, experiment.max_metal_vram))
                    else:
                        if self.init_parameters[i]['print_on']:
                            print_vram_usage(tag=f"Experiment {i+1} 训练峰值", peak=(experiment.max_torch_vram, experiment.max_metal_vram))
                elif mode == 'evaluate':
                    eval_parameters = {k: v for k, v in self.train_parameters[i].items() if k in ['batch_size', 'seq_len', 'data_type', 'sink_padding', 'd_hidden', 'function_callable', 'lorenz_kwargs']}
                    if 'loss_type' in self.init_parameters[i]:
                        eval_parameters['loss_type'] = self.init_parameters[i]['loss_type']  # 确保评估使用与训练相同类型的损失函数
                    if eval_configs and len(eval_configs) > 0:
                        for k, current_config in enumerate(eval_configs):
                            current_eval_params = eval_parameters.copy()
                            current_eval_params.update(current_config)
                            experiment.evaluate(**current_eval_params)
                    else:  # 万一没有传入eval_configs的兜底
                        print(f"⚠️ No eval_configs provided for evaluation, using default eval parameters for Experiment {i+1}.")
                        default_config = {'eval_name': f'eval_{eval_count}'}
                        current_eval_params = eval_parameters.copy()
                        current_eval_params.update(default_config)
                        experiment.evaluate(**current_eval_params)
                        eval_count += 1
                    self.results[i] = experiment.get_results(result_set)
            experiment.offload_to_cpu()  # 训练完成后将模型和结果移回CPU，节省GPU显存
            self.experiments[i] = experiment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if not parallel:
                if self.init_parameters[i]['print_on']:
                    print_vram_usage(tag=f"Experiment {i+1} 清理后")
                    print(f"Experiment {i+1}/{self.num_experiments} completed.\n")
        if any(self.init_parameters[i]['print_on'] for i in range(self.num_experiments)):
            print_vram_usage(tag=f"本底显存检测")
        if parallel_workers == 1:
            for i in range(self.num_experiments):
                single_train(i, parallel=False)
        elif isinstance(parallel_workers, int) and parallel_workers > 1:
            if any(self.init_parameters[i]['print_on'] for i in range(self.num_experiments)):
                print(f"Running {self.num_experiments} experiments in parallel with {parallel_workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                executor.map(lambda i: single_train(i, parallel=True), range(self.num_experiments))
        else:
            raise ValueError("parallel_workers must be a positive integer.")

        if self.results[0] is not None:
            invalid_keys = [key for key in result_set if key not in self.results[0]]
            if invalid_keys:
                print(f"\n[Final Check Warning] 以下请求的指标在所有流程结束后仍未找到，请检查拼写是否正确或数据是否生成失败: \n{invalid_keys}\n")

    def plot(self, compare_experiments=True,
             subplot_shape=(1, -1), figure_size=None, suptitle='Looped Transformer Experiment Results',
             colors=['blue', 'red', 'green', 'yellow', 'cyan', 'orange', 'purple', 'brown']):
        """根据 run() 收集的数据自动渲染对比图。

        支持两类横轴与两种绑图模式：
        - ``compare_experiments=True``：所有实验的同一指标画在同一张子图横向对比；
          多指标自动拆为多张子图。
        - ``compare_experiments=False``：每个实验 × 每个 result_list 独立成子图，
          result_list 中 ``'|'`` 左侧指标绑左轴、右侧绑右轴（twinx）。

        横轴 ``'epoch'``/``'block'`` 画折线，``'experiment'`` 画柱状图。支持 baseline
        相对值（相对 baseline_index 的差值）。

        Args:
            compare_experiments (bool): True 为横向对比模式，False 为单实验独立模式。
            subplot_shape (tuple): 子图网格 (rows, cols)，-1 自动计算。
            figure_size (tuple or None): 画布尺寸（英寸）；None 时取 (8*cols, 6*rows)。
            suptitle (str or None): 整张图的顶部标题，None 不显示。
            colors (list[str]): 线/柱颜色循环列表。

        Raises:
            ValueError: subplot_shape 两个维度都为 -1，或格子数不足以容纳所有图。
        """
        # 画图
        num_plots = 0
        for item in self.result_lists:
            x_type = item[1]
            clean_list = [m for m in item[0] if m != '|']
            if x_type in ['epoch', 'block']:
                if compare_experiments:
                    num_plots += len(clean_list)
                else:
                    num_plots += self.num_experiments
            elif x_type == 'experiment':
                num_plots += 1
        if subplot_shape == (-1, -1):
            raise ValueError("Invalid subplot_shape: both dimensions cannot be -1. Please specify at least one dimension or provide a valid shape.")
        if subplot_shape[1] == -1:
            subplot_shape = (subplot_shape[0], math.ceil(num_plots / subplot_shape[0]))  # 自动计算列数
        elif subplot_shape[0] == -1:
            subplot_shape = (math.ceil(num_plots / subplot_shape[1]), subplot_shape[1])  # 自动计算行数
        if subplot_shape[0] * subplot_shape[1] < num_plots:
            raise ValueError(f"subplot_shape {subplot_shape} is too small for the number of plots {num_plots}. Please increase the dimensions.")
        if figure_size is None:
            figure_size = (8 * subplot_shape[1], 6 * subplot_shape[0])

        if self.results[0] is None:
            print("No results to plot. Please run the experiments first.")
            return
        results = self.results
        fig, axs = plt.subplots(*subplot_shape, figsize=figure_size)
        axs_flat = np.atleast_1d(axs).flatten()  # 将 axs 转换为一维数组，方便索引
        index = 0
        for item in self.result_lists:
            result_list = item[0]  # 需要展示的结果名称列表
            x = item[1]            # 横坐标类型 ('epoch' 或 'experiment')
            baseline_index = item[2] if len(item) > 2 else None  # 基线索引（可选），表示用于计算相对值的基线实验索引
            baseline_name = None
            if x in ['epoch', 'block']:  # 表示画横坐标是epoch或block的曲线
                if compare_experiments:
                    clean_result_list = [m for m in result_list if m != '|']
                    for metric in clean_result_list:
                        if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                            baseline_name = self.init_parameters[baseline_index].get('experiment_name', f'Experiment {baseline_index+1}')
                            baseline_data = np.array(results[baseline_index][metric])
                        else:
                            baseline_name = None
                            baseline_data = None
                        for i in range(self.num_experiments):
                            exp_name = self.init_parameters[i].get('experiment_name', f'Experiment {i+1}')
                            y_data = np.array(results[i][metric])
                            label_name = exp_name  # 专门给图例起名字，保护原变量
                            if baseline_name is not None:
                                min_length = min(len(y_data), len(baseline_data))
                                y_data = y_data[:min_length] - baseline_data[:min_length]  # 计算相对于基线的差值
                                if i == baseline_index:
                                    label_name += ' (Baseline)'
                            axs_flat[index].plot(y_data, label=label_name, color=colors[i % len(colors)])
                        axs_flat[index].set_xlabel('Epoch' if x == 'epoch' else 'Block (Depth)')
                        if baseline_name is not None:
                            # 加上 \n 换行，防止 ylabel 太长挤出画布边界
                            axs_flat[index].set_ylabel(f"{metric}\n(relative to {baseline_name})")
                        else:
                            axs_flat[index].set_ylabel(metric)
                        axs_flat[index].tick_params(axis='y')
                        axs_flat[index].set_title('Comparison of ' + metric)
                        axs_flat[index].legend(loc='upper right')
                        axs_flat[index].grid(True)
                        # 为顶部留出 35% 的空间防止遮挡图例
                        y_min, y_max = axs_flat[index].get_ylim()
                        axs_flat[index].set_ylim(y_min, y_max + 0.35 * (y_max - y_min))
                        index += 1
                else:
                    if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                        baseline_name = self.init_parameters[baseline_index].get('experiment_name', f'Experiment {baseline_index+1}')
                        baseline_data_dict = {m: np.array(results[baseline_index][m]) for m in result_list if m != '|'}
                    else:
                        baseline_name = None
                        baseline_data_dict = None
                    for i in range(self.num_experiments):
                        color_index = 0
                        exp_name = self.init_parameters[i].get('experiment_name', f'Experiment {i+1}')
                        if '|' in result_list:
                            split_idx = result_list.index('|')
                            clean_result_list = [metric for metric in result_list if metric != '|']
                        else:
                            split_idx = len(result_list)
                            clean_result_list = result_list
                        num_metrics = len(clean_result_list)
                        for j in range(split_idx):
                            metric_j = clean_result_list[j]
                            y_data = np.array(results[i][metric_j])
                            label_name = metric_j
                            if baseline_name is not None:
                                baseline_data = baseline_data_dict[metric_j]
                                min_length = min(len(y_data), len(baseline_data))
                                y_data = y_data[:min_length] - baseline_data[:min_length]  # 计算相对于基线的差值
                                if i == baseline_index:
                                    label_name += ' (Baseline)'
                            axs_flat[index].plot(y_data, label=label_name, color=colors[color_index % len(colors)])
                            color_index += 1
                        axs_flat[index].set_xlabel('Epoch' if x == 'epoch' else 'Block (Depth)')
                        ylabel_left = '\n'.join(clean_result_list[:split_idx])
                        if baseline_name is not None:
                            ylabel_left += f"\n(relative to {baseline_name})"
                        axs_flat[index].set_ylabel(ylabel_left, color=colors[0])
                        axs_flat[index].tick_params(axis='y', labelcolor=colors[0])
                        if num_metrics > split_idx:
                            ax_twin = axs_flat[index].twinx()  # 创建共享x轴但独立y轴的第二个坐标轴
                            primary_color = colors[split_idx % len(colors)]
                            for j in range(split_idx, num_metrics):
                                metric_j = clean_result_list[j]
                                y_data = np.array(results[i][metric_j])
                                label_name = metric_j
                                if baseline_name is not None:
                                    baseline_data = baseline_data_dict[metric_j]
                                    min_length = min(len(y_data), len(baseline_data))
                                    y_data = y_data[:min_length] - baseline_data[:min_length]  # 计算相对于基线的差值
                                    if i == baseline_index:
                                        label_name += ' (Baseline)'
                                ax_twin.plot(y_data, label=label_name, color=colors[color_index % len(colors)])
                                color_index += 1
                            ax_twin.set_ylabel('\n'.join(clean_result_list[split_idx:]), color=primary_color)
                            ax_twin.tick_params(axis='y', labelcolor=primary_color)
                            ax_twin.legend(loc='upper right')
                        axs_flat[index].set_title(exp_name + ' - ' + ' and\n'.join(clean_result_list))
                        axs_flat[index].legend(loc='upper left')
                        axs_flat[index].grid(True)
                        # 为顶部留出 35% 的空间防止遮挡图例
                        y1_min, y1_max = axs_flat[index].get_ylim()
                        axs_flat[index].set_ylim(y1_min, y1_max + 0.35 * (y1_max - y1_min))
                        if num_metrics > split_idx:
                            y2_min, y2_max = ax_twin.get_ylim()
                            ax_twin.set_ylim(y2_min, y2_max + 0.35 * (y2_max - y2_min))
                        index += 1
            elif x == 'experiment':  # 表示画不同实验对比的柱状图
                exp_names = [self.init_parameters[i].get('experiment_name', f'Experiment {i+1}') for i in range(self.num_experiments)]
                if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                    baseline_name = self.init_parameters[baseline_index].get('experiment_name', f'Experiment {baseline_index+1}')
                    exp_names[baseline_index] += '\n(Baseline)'

                x_pos = np.arange(len(exp_names))
                if '|' in result_list:
                    split_idx = result_list.index('|')
                    clean_result_list = [metric for metric in result_list if metric != '|']
                else:
                    split_idx = len(result_list)
                    clean_result_list = result_list
                num_metrics = len(clean_result_list)
                width = 0.8 / num_metrics  # 确保所有柱子加起来不超过 0.8 的宽度，留下 0.2 的组间空白
                color_index = 0
                # 左侧坐标轴
                for j in range(split_idx):
                    metric_j = clean_result_list[j]
                    y_data_j = [results[i][metric_j] for i in range(self.num_experiments)]
                    if baseline_name is not None:
                        base_val = y_data_j[baseline_index]
                        y_data_j = [val - base_val for val in y_data_j]
                    offset_j = (j - num_metrics / 2 + 0.5) * width
                    axs_flat[index].bar(x_pos + offset_j, y_data_j, width, label=metric_j, color=colors[color_index % len(colors)])
                    color_index += 1
                ylabel_left = '\n'.join(clean_result_list[:split_idx])
                if baseline_name is not None:
                    ylabel_left += f"\n(relative to {baseline_name})"
                axs_flat[index].set_ylabel(ylabel_left, color=colors[0])
                axs_flat[index].tick_params(axis='y', labelcolor=colors[0])

                if num_metrics > split_idx:
                    ax_twin = axs_flat[index].twinx()  # 创建共享 X 轴但独立 Y 轴的坐标系
                    primary_color = colors[split_idx % len(colors)]
                    for j in range(split_idx, num_metrics):
                        metric_j = clean_result_list[j]
                        y_data_j = [results[i][metric_j] for i in range(self.num_experiments)]
                        if baseline_name is not None:
                            base_val = y_data_j[baseline_index]
                            y_data_j = [val - base_val for val in y_data_j]

                        offset_j = (j - num_metrics / 2 + 0.5) * width
                        ax_twin.bar(x_pos + offset_j, y_data_j, width, label=metric_j, color=colors[color_index % len(colors)])
                        color_index += 1
                    ylabel_right = '\n'.join(clean_result_list[split_idx:])
                    if baseline_name is not None:
                        ylabel_right += f"\n(relative to {baseline_name})"
                    ax_twin.set_ylabel(ylabel_right, color=primary_color)
                    ax_twin.tick_params(axis='y', labelcolor=primary_color)
                    ax_twin.legend(loc='upper right')
                    axs_flat[index].legend(loc='upper left')
                else:
                    axs_flat[index].legend(loc='best')
                y1_min, y1_max = axs_flat[index].get_ylim()
                pos1 = max(y1_max, 0)
                neg1 = max(-y1_min, 0)
                s1 = max(pos1, neg1) if max(pos1, neg1) > 0 else 1.0
                pos_norm1 = pos1 / s1
                neg_norm1 = neg1 / s1
                if num_metrics > split_idx:
                    y2_min, y2_max = ax_twin.get_ylim()
                    pos2 = max(y2_max, 0)
                    neg2 = max(-y2_min, 0)
                    s2 = max(pos2, neg2) if max(pos2, neg2) > 0 else 1.0
                    pos_norm2 = pos2 / s2
                    neg_norm2 = neg2 / s2
                    max_pos_norm = max(pos_norm1, pos_norm2)
                    max_neg_norm = max(neg_norm1, neg_norm2)
                else:
                    max_pos_norm = pos_norm1
                    max_neg_norm = neg_norm1
                    s2 = 1.0  # 占位
                final_pos_norm = max(max_pos_norm * 1.35, 0.35)
                needs_zero_alignment = (baseline_name is not None) or (y1_min < 0) or (num_metrics > split_idx and y2_min < 0)
                if needs_zero_alignment:
                    # 按照统一计算好的归一化比例，乘以各自的真实尺度
                    axs_flat[index].set_ylim(-max_neg_norm * s1, final_pos_norm * s1)
                    if num_metrics > split_idx:
                        ax_twin.set_ylim(-max_neg_norm * s2, final_pos_norm * s2)
                else:
                    axs_flat[index].set_ylim(0, final_pos_norm * s1)
                    if num_metrics > split_idx:
                        ax_twin.set_ylim(0, final_pos_norm * s2)
                axs_flat[index].axhline(0, color='gray', linewidth=1, linestyle='-', zorder=0)  # 画一条贯穿整个图表的辅助零线，充当视觉上的"地平线"
                axs_flat[index].set_xticks(x_pos)
                axs_flat[index].set_xticklabels(exp_names, rotation=45, ha='right')
                axs_flat[index].set_title(f"Comparison of {' and\n'.join(clean_result_list)}")
                axs_flat[index].grid(axis='y', linestyle='--', alpha=0.7)
                index += 1
        clear_lorenz_cache()  # 清理洛伦兹系统数据缓存，释放内存
        # 如果绘制的图的数量少于子图的总数量，则隐藏多余的子图
        for i in range(index, len(axs_flat)):
            axs_flat[i].set_visible(False)
        if suptitle is not None:
            plt.suptitle(suptitle + "\n")
        plt.tight_layout()
        plt.show()
