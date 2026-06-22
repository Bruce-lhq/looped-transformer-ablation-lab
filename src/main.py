import sys
from pathlib import Path

# looped_transformer 包在项目根目录（src/ 的上一级），需把它加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looped_transformer import ExperimentTable

if __name__ == '__main__':
    # 位置编码对比实验
    table = ExperimentTable(params_groups=[
        {'pe_type': ['learned_ape'], 'experiment_name': 'Learned APE'},
        {'pe_type': ['alibi'],       'experiment_name': 'ALiBi'},
        {'pe_type': ['rope'],        'experiment_name': 'RoPE'},
    ])
    table.run(result_lists=[(['loss_history'], 'epoch')], parallel_workers=1)
    table.plot(figure_size=(8, 6))
