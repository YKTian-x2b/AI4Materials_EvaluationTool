import subprocess
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from nvitop import ResourceMetricCollector, collect_in_background

from Utils import MetricsForPrediction
from Utils import GenericMetrics
import os
import sys
# 调整工作目录
current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(current_dir)
sys.path.insert(0, current_dir)


def run():
    pyFile = 'symmetry_prediction_my.py'
    command = 'python ' + pyFile
    output_dir_ = current_dir + "res/"
    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)
    # getxPUInfo(command, output_dir_)

    df = pd.read_csv(output_dir_ + 'pred_result_test.csv')
    crystal_system_pred = df['crystal_system_pred']
    crystal_system_true= df['crystal_system_true']
    sg_num_pred = df['sg_num_pred']
    sg_num_true = df['sg_num_true']
    # 这里提供了另一个思路，就是我们在运行过程中把结果写入csv文件，eval的时候直接读文件，而不是再跑一次
    getEvalMetrics(crystal_system_true, crystal_system_pred, output_dir_)
    getEvalMetrics(sg_num_true, sg_num_pred, output_dir_)


def getxPUInfo(command, output_dir):
    logFile = 'res_log.txt'
    errFile = 'err_log.txt'
    df = pd.DataFrame()
    collector = ResourceMetricCollector(root_pids={1}, interval=2.0)
    with collector(tag='resources'):
        log = open(output_dir + logFile, 'w')
        err = open(output_dir + errFile, 'w')
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
    df.to_csv(output_dir + 'metrics.csv', index=False)


def getEvalMetrics(targets, predictions, output_dir):
    testAccuracy = MetricsForPrediction.getAccuracy(targets, predictions)
    testRecall = MetricsForPrediction.getRecall(targets, predictions, 'macro')
    testPrecision = MetricsForPrediction.getPrecision(targets, predictions, 'macro')
    testF1Score = MetricsForPrediction.getF1Score(targets, predictions, 'macro')
    testROCandAUC = MetricsForPrediction.getROCandAUC(targets, predictions)

    msg = f"testAccuracy={testAccuracy}; testRecall={testRecall}; testPrecision={testPrecision}; "
    msg += f"testF1Score={testF1Score}; testROCandAUC={testROCandAUC}" + "\n"
    print(msg)
    with open(output_dir + 'evalMetrics.txt', 'a') as file:
        file.write(msg)
