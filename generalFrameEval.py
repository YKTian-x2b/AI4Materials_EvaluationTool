import subprocess
import pandas as pd
import torch
from torch.nn import nn
from datetime import datetime
from nvitop import ResourceMetricCollector, collect_in_background

from Utils import MetricsForGeneration
from Utils import GenericMetrics
import os
import sys

# 调整工作目录
current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(current_dir)
sys.path.insert(0, current_dir)


# def util():
#     pyFile = 'train.py'
#     resPath = 'result/'
#     dataset = datasetName
#     command = 'python ' + pyFile + ' --result_path ' + resPath + \
#               ' --dataset ' + dataset
#
#     train_data_path = os.path.join('dataConfig', dataset, 'train.pt')
#     if not os.path.isfile(train_data_path):
#         train_data_path = os.path.join('dataConfig', dataset, 'train.csv')
#     test_data_path = os.path.join('dataConfig', dataset, 'test.pt')
#     if not os.path.isfile(test_data_path):
#         test_data_path = os.path.join('dataConfig', dataset, 'test.csv')
#     # val_data_path = os.path.join('dataConfig', dataset, 'val.pt')
#     # if not os.path.isfile(val_data_path):
#     #     val_data_path = os.path.join('dataConfig', dataset, 'val.csv')
#
#     score_norm_path = os.path.join('dataConfig', dataset, 'score_norm.txt')
#     os.chdir(current_dir)
#     if dataset == 'perov_5':
#         from config.perov_5_config_dict import conf
#     elif dataset == 'carbon_24':
#         from config.carbon_24_config_dict import conf
#     else:
#         from config.mp_20_config_dict import conf
#
#     model = MatGen(**conf['model'], score_norm=np.loadtxt(score_norm_path))
    # print("executing getFLOPSandParams ...")

    #
    # print("executing generate and eval...")
    # model_path = 'result/model_699.pth'
    # num_gen = 10000
    # if not os.path.isfile(model_path):
    #     raise Exception("Model path is not exist!")
    # generate(model_path, test_data_path, train_data_path, conf, score_norm_path, num_gen)


def run(command, model, dataloader):
    resDir = "res/"
    getxPUInfo(command, resDir)
    getxPUInfoList(command, resDir)

    assert (isinstance(model, nn.Module))
    getFLOPSandParams(model, dataloader, resDir)


def getxPUInfo(command, resDir):
    logFile = resDir + 'xPUInfo_res_log.txt'
    errFile = resDir + 'xPUInfo_err_log.txt'
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
    df.to_csv(resDir + 'metrics.csv', index=False)


def getxPUInfoList(command, resDir):
    resFile = resDir + 'xPUInfoList_res_log.txt'
    errFile = resDir + 'xPUInfoList_err_log.txt'
    res = open(current_dir + resFile, 'w')
    err = open(current_dir + errFile, 'w')

    def on_collect(metrics):
        if res.closed:
            return False
        with open(resDir + 'metrics.csv', 'a', newline='') as file:
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
            csv_string = df_metrics.to_csv(index=False, header=False)
            if os.path.getsize(resDir + 'metrics.csv') == 0:
                csv_string = df_metrics.to_csv(index=False)
            file.write(csv_string)
        return True

    def on_stop(collector):
        if not res.closed:
            res.close()
        print('collection end!')

    collect_in_background(
        on_collect,
        ResourceMetricCollector(root_pids={1}),
        interval=2.0,
        on_stop=on_stop,
    )
    process = subprocess.Popen(command, shell=True, stdout=res, stderr=err)
    print("running...")
    process.wait()
    res.close()
    err.close()
    print("end...")


def getFLOPSandParams(model, loader, resDir):
    total_flops = 0
    total_params = 0
    for iter_num, data_batch in enumerate(loader):
        data_batch = data_batch.to("cuda")
        flops, params = GenericMetrics.getFLOPSandParams(model, data_batch)
        total_flops += flops
        total_params += params
    msg = f'total_flops: {total_flops}, total_params: {total_params}'
    print(msg)
    with open(resDir + 'flopsAndParams.txt', 'w') as paramsFile:
        paramsFile.write(msg)
