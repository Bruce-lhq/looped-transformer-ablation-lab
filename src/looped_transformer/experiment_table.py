"""多实验调度与可视化模块。

ExperimentTable 是多实验对比的核心调度器：
传入一组参数字典 → 自动跑完所有实验 → 收集指标 → 生成对比图。
支持多线程并行压测和自适应子图布局。
"""

import copy
import math
import threading
import concurrent.futures
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from .experiment import LoopedTransformerExperiment, print_vram_usage
from .parameters import default_setup


class ExperimentTable:
    """多实验调度与自动画图引擎。

    底层逻辑是"全量默认 + 局部覆写"：自动调用 default_setup() 生成默认参数，
    params_groups 中填什么就覆写什么。支持串行/并行执行、训练/评估双模式、
    自动对比绑图。

    Attributes:
        num_experiments (int): 实验总数。
        init_parameters (list[dict]): 每个实验的初始化参数。
        train_parameters (list[dict]): 每个实验的训练参数。
        result_lists (list[tuple]): 用户指定的需要收集/画图的数据配置。
        results_groups (dict): 收集到的结果，按模式分组。
        experiments (list[LoopedTransformerExperiment]): 实验对象列表（已转移至 CPU）。
    """

    def __init__(self, params_groups: list[dict], manual=None):
        """初始化实验台。

        Args:
            params_groups (list[dict]): 每个元素是一个实验的参数覆写字典。
                未指定的 key 使用 default_setup() 的默认值。
                每个字典必须使用在 init_parameters 或 train_parameters 中存在的 key。

            manual (dict or None): 全局参数覆盖，在逐实验覆写之前应用到所有实验。

        Raises:
            ValueError: 当 params_groups 中存在未知参数名时抛出。
        """
        init, train = default_setup(manual=manual)
        self.num_experiments = len(params_groups)
        self.init_parameters = [copy.deepcopy(init) for _ in range(self.num_experiments)]
        self.train_parameters = [copy.deepcopy(train) for _ in range(self.num_experiments)]
        for i, params in enumerate(params_groups):
            for key, value in params.items():
                if key in self.init_parameters[i]:
                    self.init_parameters[i][key] = value
                elif key in self.train_parameters[i]:
                    self.train_parameters[i][key] = value
                else:
                    raise ValueError(f"Unknown parameter: {key}")

    def run(self, result_lists: list[tuple[list[str], str, int]], modes=None, parallel_workers=1):
        """执行所有实验。

        根据 parallel_workers 决定串行或并行执行。并行模式下自动关闭日志输出
        以避免竞态。每个实验完成后自动 offload_to_cpu() 并清空 GPU 缓存。

        Args:
            result_lists (list[tuple]): 要收集的数据指标。
                格式：[(['指标1', '指标2'], '横轴类型'), ...]
                或 [(['指标1', '指标2'], '横轴类型', baseline_index), ...]
                横轴类型 'epoch' 表示训练过程指标，'experiment' 表示汇总指标。
                baseline_index 可选，表示用于计算相对值的基线实验索引。
            modes (list[str] or None): 运行模式，['train'] 和/或 ['evaluate']。
                默认 ['train']。
            parallel_workers (int): 并行线程数。1 为串行，>1 为多线程并行。

        Raises:
            ValueError: 当 parallel_workers 不是正整数时抛出。
        """
        if modes is None:
            modes = ['train']
        result_set = set().union(*[item[0] for item in result_lists])
        self.result_lists = result_lists
        self.results_groups = {
            'train': [None] * self.num_experiments,
            'evaluate': [None] * self.num_experiments,
        }
        self.experiments = [None] * self.num_experiments
        print_lock = threading.Lock()
        vram_printed = False

        for i in range(self.num_experiments):
            experiment = LoopedTransformerExperiment(**self.init_parameters[i])
            experiment.offload_to_cpu()
            self.experiments[i] = experiment

        def single_train(i, parallel=False):
            nonlocal vram_printed
            experiment = self.experiments[i]
            experiment.load_to_device()
            if parallel:
                self.train_parameters[i]['timing'] = False
                self.train_parameters[i]['print_every'] = None
            if parallel:
                with print_lock:
                    if not vram_printed:
                        print_vram_usage(tag="Parallel峰值显存检测")
                        vram_printed = True
            else:
                print_vram_usage(tag=f"Experiment {i+1} 训练前")
            for mode in modes:
                if mode == 'train':
                    experiment.train(**self.train_parameters[i])
                    self.results_groups['train'][i] = experiment.get_results(result_set)
                    if parallel:
                        with print_lock:
                            print_vram_usage(
                                tag=f"Experiment {i+1} 训练峰值",
                                peak=(experiment.max_torch_vram, experiment.max_metal_vram),
                            )
                    else:
                        print_vram_usage(
                            tag=f"Experiment {i+1} 训练峰值",
                            peak=(experiment.max_torch_vram, experiment.max_metal_vram),
                        )
                elif mode == 'evaluate':
                    eval_parameters = {
                        k: v for k, v in self.train_parameters[i].items()
                        if k in ['batch_size', 'seq_len', 'data_type', 'sink_padding']
                    }
                    if 'loss_type' in self.init_parameters[i]:
                        eval_parameters['loss_type'] = self.init_parameters[i]['loss_type']
                    self.results_groups['evaluate'][i] = experiment.evaluate(**eval_parameters)
            experiment.offload_to_cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if not parallel:
                print_vram_usage(tag=f"Experiment {i+1} 清理后")
                print(f"Experiment {i+1}/{self.num_experiments} completed.\n")

        print_vram_usage(tag="本底显存检测")
        if parallel_workers == 1:
            for i in range(self.num_experiments):
                single_train(i, parallel=False)
        elif isinstance(parallel_workers, int) and parallel_workers > 1:
            print(f"Running {self.num_experiments} experiments in parallel with {parallel_workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                executor.map(lambda i: single_train(i, parallel=True), range(self.num_experiments))
        else:
            raise ValueError("parallel_workers must be a positive integer.")

    def plot(self, compare_experiments=True,
             subplot_shape=(1, -1), figure_size=(15, 10),
             suptitle='Looped Transformer Experiment Results',
             colors=None):
        """根据 run() 收集的数据自动渲染对比图。

        支持两种模式：
        - compare_experiments=True：所有实验的同一指标放在同一张子图中对比。
          多指标时自动拆分为多张子图。
        - compare_experiments=False：每个实验 × 每个 result_list 独立成子图。
          多指标使用双 Y 轴（第一个指标左轴，其余右轴）。

        Args:
            compare_experiments (bool): True 为横向对比模式，
                False 为单实验独立模式。默认 True。
            subplot_shape (tuple): 子图网格布局 (rows, cols)。
                -1 表示自动计算。如 (1, -1) 表示一行排列。
                默认 (1, -1)。
            figure_size (tuple): 画布尺寸（英寸），默认 (15, 10)。
            suptitle (str or None): 整张图的顶部标题，None 不显示。
            colors (list[str] or None): 线条颜色循环列表。
                None 使用默认 8 色调色板。

        Raises:
            ValueError: subplot_shape 两个维度都是 -1 时抛出。
            ValueError: 子图格子数不足以容纳所有图表时抛出。

        Note:
            在非交互式后端（如 'Agg'）下不会调用 plt.show()，方便脚本化保存图片。
        """
        if colors is None:
            colors = ['blue', 'red', 'green', 'yellow', 'cyan', 'orange', 'purple', 'brown']

        if compare_experiments:
            num_plots = len(self.result_lists)
        else:
            num_plots = len(self.result_lists) * self.num_experiments

        if subplot_shape == (-1, -1):
            raise ValueError(
                "Invalid subplot_shape: both dimensions cannot be -1. "
                "Please specify at least one dimension or provide a valid shape."
            )
        if subplot_shape[1] == -1:
            subplot_shape = (subplot_shape[0], math.ceil(num_plots / subplot_shape[0]))
        elif subplot_shape[0] == -1:
            subplot_shape = (math.ceil(num_plots / subplot_shape[1]), subplot_shape[1])
        if subplot_shape[0] * subplot_shape[1] < num_plots:
            raise ValueError(
                f"subplot_shape {subplot_shape} is too small for the number of plots {num_plots}. "
                f"Please increase the dimensions."
            )

        if self.results_groups['train'][0] is not None:
            results = self.results_groups['train']
            fig, axs = plt.subplots(*subplot_shape, figsize=figure_size)
            axs_flat = np.atleast_1d(axs).flatten()
            index = 0
            for item in self.result_lists:
                result_list = item[0]
                x = item[1]
                baseline_index = item[2] if len(item) > 2 else None
                if x == 'epoch':
                    if compare_experiments:
                        for metric in result_list:
                            baseline_name = None
                            if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                                baseline_name = self.init_parameters[baseline_index].get('experiment_name', f'Experiment {baseline_index + 1}')
                                baseline_data = np.array(results[baseline_index][metric])
                            for i in range(self.num_experiments):
                                exp_name = self.init_parameters[i].get('experiment_name', f'Experiment {i+1}')
                                y_data = np.array(results[i][metric])
                                if baseline_name is not None:
                                    y_data = y_data - baseline_data
                                    if i == baseline_index:
                                        exp_name += ' (Baseline)'
                                axs_flat[index].plot(y_data, label=exp_name, color=colors[i % len(colors)])
                            axs_flat[index].set_xlabel('Epoch')
                            if baseline_name is not None:
                                axs_flat[index].set_ylabel(f"{metric} (relative to {baseline_name})")
                            else:
                                axs_flat[index].set_ylabel(metric)
                            axs_flat[index].tick_params(axis='y')
                            axs_flat[index].set_title('Comparison of ' + metric)
                            axs_flat[index].legend(loc='upper right')
                            axs_flat[index].grid(True)
                            index += 1
                    else:
                        if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                            baseline_name = self.init_parameters[baseline_index].get('experiment_name', f'Experiment {baseline_index + 1}')
                            baseline_data = np.array(results[baseline_index][result_list[0]])
                        for i in range(self.num_experiments):
                            color_index = 0
                            y_data = np.array(results[i][result_list[0]])
                            if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                                baseline_name = self.init_parameters[baseline_index].get('experiment_name', f'Experiment {baseline_index + 1}')
                                y_data = y_data - baseline_data
                                if i == baseline_index:
                                    exp_name_extra = ' (Baseline)'
                            axs_flat[index].plot(
                                y_data, label=result_list[0],
                                color=colors[color_index % len(colors)],
                            )
                            axs_flat[index].set_xlabel('Epoch')
                            if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                                axs_flat[index].set_ylabel(f"{result_list[0]} (relative to {baseline_name})", color=colors[color_index % len(colors)])
                            else:
                                axs_flat[index].set_ylabel(result_list[0], color=colors[color_index % len(colors)])
                            axs_flat[index].tick_params(axis='y', labelcolor=colors[color_index % len(colors)])
                            color_index += 1
                            if len(result_list) > 1:
                                ax_twin = axs_flat[index].twinx()
                                primary_color = colors[color_index % len(colors)]
                                for j in range(1, len(result_list)):
                                    y_data = np.array(results[i][result_list[j]])
                                    if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                                        baseline_data_j = np.array(results[baseline_index][result_list[j]])
                                        y_data = y_data - baseline_data_j
                                    ax_twin.plot(
                                        y_data, label=result_list[j],
                                        color=colors[color_index % len(colors)],
                                    )
                                    color_index += 1
                                if baseline_index is not None and 0 <= baseline_index < self.num_experiments:
                                    ax_twin.set_ylabel(' and '.join(result_list[1:]) + f' (relative to {baseline_name})', color=primary_color)
                                else:
                                    ax_twin.set_ylabel(' and '.join(result_list[1:]), color=primary_color)
                                ax_twin.tick_params(axis='y', labelcolor=primary_color)
                                ax_twin.legend(loc='upper right')
                            axs_flat[index].set_title(
                                self.init_parameters[i]['experiment_name'] + '-' + ' and '.join(result_list)
                            )
                            axs_flat[index].legend(loc='upper left')
                            axs_flat[index].grid(True)
                            index += 1
                elif x == 'experiment':
                    pass
            for i in range(index, len(axs_flat)):
                axs_flat[i].set_visible(False)
        if self.results_groups['evaluate'][0] is not None:
            pass
        if suptitle is not None:
            plt.suptitle(suptitle)
        plt.tight_layout()
        import matplotlib
        if matplotlib.is_interactive():
            plt.show()
