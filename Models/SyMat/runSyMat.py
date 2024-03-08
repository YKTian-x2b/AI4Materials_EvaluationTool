import subprocess
import time
from datetime import datetime
from Utils import GenericMetrics
from Utils import MetricsForPrediction

def run():
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
    GenericMetrics.getCPUInfo()
    GenericMetrics.getProcessGPUInfo(process.pid)

    print("model is running...")
    process.wait()

    GenericMetrics.getProcessElapsedTime(process.pid)
    endTime = datetime.now()
    elapsedTime = endTime - startTime
    print(f"Execution time: {elapsedTime}")


def generate():
    print('generate')