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
