import subprocess
import time
from datetime import datetime
import pandas as pd
from nvitop import ResourceMetricCollector
from Utils import GenericMetrics
from Utils import MetricsForPrediction


def run2():
    pyFile = 'train.py'
    resPath = 'result/'
    dataset = 'perov_5'
    logFile = dataset + '_res_log.txt'
    command = ['python', pyFile,
               '--result_path', resPath,
               '--dataset', dataset,
               '>', logFile, '2>&1']

    startTime = datetime.now()

    process = subprocess.Popen(command, shell=True)

    time.sleep(5)
    GenericMetrics.getInfo()
    print("getProcessGPUInfo:")
    GenericMetrics.getProcessGPUInfo(process.pid)

    print("model is running...")
    process.wait()

    GenericMetrics.getProcessElapsedTime(process.pid)
    endTime = datetime.now()
    elapsedTime = endTime - startTime
    print(f"Execution time: {elapsedTime}")


def run():
    pyFile = 'train.py'
    resPath = 'result/'
    dataset = 'perov_5'
    logFile = dataset + '_res_log.txt'
    command = ['python', pyFile,
               '--result_path', resPath,
               '--dataset', dataset,
               '>', logFile, '2>&1']

    df = pd.DataFrame()
    collector = ResourceMetricCollector(root_pids={1}, interval=2.0)
    with collector(tag='resources'):
        process = subprocess.Popen(command, shell=True)
        process.wait()
        metrics = collector.collect()
        df_metrics = pd.DataFrame.from_records(metrics, index=[len(df)])
        df = pd.concat([df, df_metrics], ignore_index=True)

    df.insert(0, 'time', df['resources/timestamp'].map(datetime.fromtimestamp))
    df.to_csv('metrics.csv', index=False)


def generate():
    print('generate')