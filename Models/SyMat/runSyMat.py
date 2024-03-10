import os
import subprocess
import sys

import pandas as pd
import numpy as np
from datetime import datetime
from nvitop import ResourceMetricCollector

from Utils import GenericMetrics
from Utils import MetricsForPrediction

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def run():
    os.chdir(current_dir)
    sys.path.insert(0, current_dir)

    pyFile = 'train.py'
    resPath = 'result/'
    dataset = 'perov_5'
    command = 'python ' + pyFile + ' --result_path ' + resPath + \
              ' --dataset ' + dataset
    # getXPUInfo(command, dataset)
    getFLOPSandParams(dataset)


def getXPUInfo(command, dataset):
    logFile = dataset + '_res_log.txt'
    errFile = dataset + '_err_log.txt'
    df = pd.DataFrame()
    collector = ResourceMetricCollector(root_pids={1}, interval=2.0)
    with collector(tag='resources'):
        log = open(current_dir + logFile, 'w')
        err = open(current_dir + errFile, 'w')
        process = subprocess.Popen(command, shell=True, stdout=log, stderr=err)
        print("running...")
        process.wait()
        metrics = collector.collect()
        df_metrics = pd.DataFrame.from_records(metrics, index=[len(df)])
        df = pd.concat([df, df_metrics], ignore_index=True)
        log.close()
        err.close()
    print("end...")
    df.insert(0, 'time', df['resources/timestamp'].map(datetime.fromtimestamp))
    df.to_csv('metrics.csv', index=False)


def getFLOPSandParams(dataset):
    from dataset import MatDataset
    from model import MatGen
    print("executing getFLOPSandParams ...")
    train_data_path = os.path.join('dataConfig', dataset, 'train.pt')
    if not os.path.isfile(train_data_path):
        train_data_path = os.path.join('dataConfig', dataset, 'train.csv')

    val_data_path = os.path.join('dataConfig', dataset, 'val.pt')
    if not os.path.isfile(val_data_path):
        val_data_path = os.path.join('dataConfig', dataset, 'val.csv')

    score_norm_path = os.path.join('dataConfig', dataset, 'score_norm.txt')
    os.chdir(current_dir)
    if dataset == 'perov_5':
        from config.perov_5_config_dict import conf
    elif dataset == 'carbon_24':
        from config.carbon_24_config_dict import conf
    else:
        from config.mp_20_config_dict import conf
    model = MatGen(**conf['model'], score_norm=np.loadtxt(score_norm_path))
    input_data = MatDataset(val_data_path, **conf['dataConfig'])
    GenericMetrics.getFLOPSandParams(model, input_data)


def generate():
    print('generate')
