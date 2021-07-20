import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_plots(results_individual, dir_path, models):
    _row_list_ind = []
    splits = []
    for result_key, result in results_individual.items():
        for split_key, split_result in result.items():
            if split_key == 'random_individual':
                for _i, split_result_inner in enumerate(split_result):
                    split_result_inner: dict
                    split_key_inner = split_key + '_' + str(_i)
                    splits.append(split_key_inner)
                    for model_key, model_results in split_result_inner.items():
                        for metric_key, metric_result in model_results.items():
                            _row_list_ind.append([result_key, split_key_inner, model_key, metric_key, metric_result])
            else:
                for model_key, model_results in split_result.items():
                    for metric_key, metric_result in model_results.items():
                        _row_list_ind.append([result_key, split_key, model_key, metric_key, metric_result])

    data_ind = pd.DataFrame(_row_list_ind, columns=['dataset', 'split', 'model', 'metric', 'result'])

    # Styling parameters

    bar_width = 0.25

    datasets = list(results_individual.keys())
    metrics = ['ACC', 'PRE', 'REC']

    for dataset in datasets:
        coherent_scores = [data_ind.loc[(data_ind['dataset'] == dataset) &
                                        (data_ind['split'] == 'coherent') &
                                        (data_ind['model'].isin(models)) &
                                        (data_ind['metric'] == _metric)] for _metric in metrics]

        y_pos = np.arange(len(coherent_scores[0]))

        fig, ax = plt.subplots(figsize=([6.4, 3]))

        axes = plt.gca()
        axes.set_ylim([0.5, 1.005])

        ps = []
        for p_i in range(len(coherent_scores)):
            p = ax.bar(y_pos + bar_width * p_i, coherent_scores[p_i]['result'], bar_width, bottom=0)
            ps.append(p)

        ax.legend((_p[0] for _p in ps), ('Accuracy', 'Precision', 'Recall'), loc='lower left')

        random_scores = [[data_ind.loc[(data_ind['dataset'] == dataset) &
                                       (data_ind['split'].isin(splits)) &
                                       (data_ind['model'] == _model) &
                                       (data_ind['metric'] == _metric)] for _model in models] for _metric in metrics]

        _random_scores_means = []
        _random_scores_stds = []
        for _by_metric in random_scores:
            _metric_means = []
            _metric_stds = []
            for _by_model in _by_metric:
                _metric_means.append(np.mean(_by_model['result']))
                _metric_stds.append(np.std(_by_model['result']))
            _random_scores_means.append(_metric_means)
            _random_scores_stds.append(_metric_stds)

        y_pos = np.arange(len(_random_scores_means[0]))

        ps = []
        for p_i in range(len(_random_scores_stds)):
            p = ax.errorbar(y_pos + bar_width * p_i, _random_scores_means[p_i], _random_scores_stds[p_i],
                            linestyle='None', marker='_', color='black')
            ps.append(p)

        ax.set_xticks(y_pos + bar_width)
        ax.set_xticklabels(tuple(models))

        plt.savefig(dir_path + 'plot-' + dataset + '-comparison', bbox_inches='tight')
