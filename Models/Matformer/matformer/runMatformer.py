import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
import pandas as pd
import subprocess
import torch
from datetime import datetime
from Utils import GenericMetrics
from Utils import MetricsForPrediction
from nvitop import ResourceMetricCollector


def run():
    # 调整工作目录
    os.chdir(current_dir)
    sys.path.insert(0, current_dir)
    device = torch.device("cuda")

    pyFile = 'train_mp.py'
    command = 'python ' + pyFile

    output_dir_ = current_dir + "matformer_mp_bulk"
    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)
    # getxPUInfo(command, output_dir_)

    from runMatformerUtil import get_prop_model_config
    # ["mu_b", "elastic anisotropy"]
    props = ["e_form", "gap pbe", "bulk modulus", "shear modulus"]
    mp_id_list_ = "bulk"
    use_save_ = True
    line_graph = True
    config = get_prop_model_config(learning_rate=0.001, name="matformer", dataset="megnet",
                                   prop=props[2], pyg_input=True, n_epochs=2, batch_size=64,
                                   use_lattice=True, output_dir=output_dir_, use_angle=False,
                                   save_dataloader=False, use_save=use_save_, mp_id_list=mp_id_list_)
    # getFLOPSandParams(device, config, use_save_, mp_id_list_, line_graph)

    getEvalMetrics(config, use_save_, mp_id_list_)


def getxPUInfo(command, output_dir):
    logFile = '/res_log.txt'
    errFile = '/err_log.txt'
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
    df.to_csv(output_dir + '/metrics.csv', index=False)


def getFLOPSandParams(device, config, use_save, mp_id_list, line_graph):
    from runMatformerUtil import get_train_val_loaders
    from pyg_att import Matformer

    (train_loader, _, _, _, _, _) = get_train_val_loaders(config, use_save, mp_id_list, line_graph)
    model = Matformer(config)
    model.to(device)
    total_flops = 0
    total_params = 0
    for iter_num, data_batch in enumerate(train_loader):
        data_batch = data_batch.to("cuda")
        flops, params = GenericMetrics.getFLOPSandParams(model, data_batch)
        total_flops += flops
        total_params += params
    print(f'total_flops: {total_flops}, total_params: {total_params}')


def getEvalMetrics(config, use_save, mp_id_list):
    from train import train_dgl

    _, targets, predictions = train_dgl(config, test_only=True,
                                             use_save=use_save,
                                             mp_id_list=mp_id_list)
    testMAE = MetricsForPrediction.getMAE(targets, predictions)
    testMSE = MetricsForPrediction.getMSE(targets, predictions)
    testRMSE = MetricsForPrediction.getRMSE(targets, predictions)
    print("testMAE=", testMAE, "; testMSE=", testMSE, "; testRMSE=", testRMSE)

