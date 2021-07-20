import logging

import os
import sys
from datetime import datetime
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

from data import kdd as kdd, unsw_nb15 as unsw, nsl_kdd as nslkdd  # ,\
# cicids2017 as cicids2017, gure_kdd_6_percent as gurekdd

import models
from visualization import create_plots


def average_random_results(_results: list):
    averages = {}
    for model_type in _results[0].keys():
        averages[model_type] = {}
        for metric_name in _results[0][model_type].keys():
            averages[model_type][metric_name] = 0
            for _result in _results:
                averages[model_type][metric_name] += _result[model_type][metric_name]
            averages[model_type][metric_name] /= len(_results)
    return averages


# Configure logging
if not os.path.exists('./logs/'):
    os.makedirs('./logs/')
log_level = 'INFO'
log_file_path = './logs/log_'+datetime.now().strftime("%y-%m-%d_%Hh-%Mm-%Ss")+'.log'

file_handler = RotatingFileHandler(log_file_path)
stream_handler = StreamHandler(sys.stdout)
logging.basicConfig(**{'level': log_level,
                       'format': '[{levelname:1.1}] {asctime} {message}',
                       'style': '{',
                       'datefmt': '%Y-%m-%d %H:%M:%S',
                       'handlers': [stream_handler, file_handler]})

stream_handler.setLevel(log_level.upper())
file_handler.setLevel(log_level.upper())
logging.logThreads = 0
logging.logProcesses = 0
logging._srcfile = None

error_metrics = ['FPR']

num_random_iterations = 5

model_list = [
    # 'Naive Bayes',
    'Logistic Regression',
    'Decision Tree',
    'AdaBoost',
    'Random Forest'
]

datasets_dict = {
    'nsl-kdd': nslkdd,
    # 'cicids2017': cicids2017,
    # 'gure-kdd': gurekdd,
    'kdd': kdd,
    'unsw': unsw
}

try:
    results = {}

    for dataset_name, dataset_object in datasets_dict.items():
        logging.info('{:_^128}'.format(f"Train & Evaluate Models on Coherent Split of {dataset_name} Dataset"))
        results[dataset_name] = {}
        results[dataset_name]['coherent'] = models.train_and_test(*dataset_object.get_coherent_split(),
                                                                  model_list=model_list)
        logging.info(results[dataset_name]['coherent'])

        logging.info('{:_^128}'.format(f"Train & Evaluate Models on Random Split of {dataset_name} Dataset"))
        random_results = []
        for i in range(0, num_random_iterations):
            random_result = models.train_and_test(*dataset_object.get_random_split(),
                                                  model_list=model_list)
            logging.info(random_result)
            random_results.append(random_result)
        results[dataset_name]['random_individual'] = random_results
        results[dataset_name]['random'] = average_random_results(random_results)
        logging.info(results[dataset_name]['random'])

    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    plots_path = './plots/' + datetime.now().strftime("%d-%m-%y_%Hh-%Mm-%Ss") + '/'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    create_plots(results, plots_path, model_list)

except Exception as exception:
    logging.exception(exception)
    raise exception
