import subprocess
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from nvitop import ResourceMetricCollector, collect_in_background

import os
import sys
# 调整工作目录
current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(current_dir)
sys.path.insert(0, current_dir)


def run(runPathIdx):
    # python CGCNN_torch/main.py data/root_dir_1000 > res/torch_res.txt
    # python CGCNN_paddle/main.py data/root_dir_1000 > res/paddle_res.txt

    pyFileList = ['CGCNN_torch/main.py', 'CGCNN_paddle/main.py']
    resFileList = ['res/TorchRes/', 'res/PaddleRes/']
    dataPath = 'data/root_dir_1000'

    command = 'python ' + pyFileList[runPathIdx] + ' ' + dataPath
    getxPUInfoList(command, resFileList[runPathIdx])


def getxPUInfoList(command, resPath):
    resDir = resPath
    resFile = resDir + 'res_log.txt'
    errFile = resDir + 'err_log.txt'
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