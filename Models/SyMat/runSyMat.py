import subprocess
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from nvitop import ResourceMetricCollector, collect_in_background
from torch_geometric.data import DataLoader

from Utils import MetricsForGeneration
from Utils import GenericMetrics
import os
import sys
# 调整工作目录
current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(current_dir)
sys.path.insert(0, current_dir)
from runner import Runner
from utils import get_structure
from dataset import MatDataset
from model import MatGen


def run(datasetName='perov_5'):
    # getxPUInfo
    pyFile = 'train.py'
    resPath = 'result/'
    dataset = datasetName
    command = 'python ' + pyFile + ' --result_path ' + resPath + \
              ' --dataset ' + dataset
    # getxPUInfo(command, dataset)
    # getxPUInfoList(command, dataset)

    train_data_path = os.path.join('dataConfig', dataset, 'train.pt')
    if not os.path.isfile(train_data_path):
        train_data_path = os.path.join('dataConfig', dataset, 'train.csv')
    test_data_path = os.path.join('dataConfig', dataset, 'test.pt')
    if not os.path.isfile(test_data_path):
        test_data_path = os.path.join('dataConfig', dataset, 'test.csv')
    # val_data_path = os.path.join('dataConfig', dataset, 'val.pt')
    # if not os.path.isfile(val_data_path):
    #     val_data_path = os.path.join('dataConfig', dataset, 'val.csv')

    score_norm_path = os.path.join('dataConfig', dataset, 'score_norm.txt')
    os.chdir(current_dir)
    if dataset == 'perov_5':
        from config.perov_5_config_dict import conf
    elif dataset == 'carbon_24':
        from config.carbon_24_config_dict import conf
    else:
        from config.mp_20_config_dict import conf

    model = MatGen(**conf['model'], score_norm=np.loadtxt(score_norm_path))
    # print("executing getFLOPSandParams ...")
    # getFLOPSandParams(model, train_data_path, conf)

    print("executing generate and eval...")
    model_path = 'result/model_699.pth'
    num_gen = 10
    if not os.path.isfile(model_path):
        raise Exception("Model path is not exist!")
    generate(model_path, test_data_path, train_data_path, conf, score_norm_path, num_gen)


def getxPUInfo(command, dataset):
    resDir = 'tmpRes/'
    logFile = resDir + dataset + '_res_log.txt'
    errFile = resDir + dataset + '_err_log.txt'
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


def getxPUInfoList(command, dataset):
    resDir = 'tmpRes/'
    resFile = resDir + dataset + '_res_log.txt'
    errFile = resDir + dataset + '_err_log.txt'
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



def getFLOPSandParams(model, train_data_path, conf):
    dataset = MatDataset(train_data_path, **conf['dataConfig'])
    loader = DataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    total_flops = 0
    total_params = 0
    for iter_num, data_batch in enumerate(loader):
        data_batch = data_batch.to("cuda")
        flops, params = GenericMetrics.getFLOPSandParams(model, data_batch)
        total_flops += flops
        total_params += params
    msg = f'total_flops: {total_flops}, total_params: {total_params}'
    print(msg)
    with open('result/flopsAndParams.txt', 'w') as paramsFile:
        paramsFile.write(msg)


def generate(model_path, test_data_path, train_data_path, conf, score_norm_path, num_gen):
    # pyFile = 'generate.py'
    # modelPath = 'result/model_699.pth'
    # dataset = 'perov_5'
    # numGen = 100
    # # python generate.py --model_path result/model_699.pth --dataset perov_5 --num_gen 100
    # command = 'python ' + pyFile + ' --model_path ' + modelPath + \
    #           ' --dataset ' + dataset + '--num_gen' + numGen

    dataset = MatDataset(test_data_path, prop_name=conf['dataConfig']['prop_name'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    gt_atom_types_list, gt_lengths_list, gt_angles_list, gt_frac_coords_list = [], [], [], []
    for iter_num, data_batch in enumerate(loader):
        atom_types, lengths, angles, frac_coords = data_batch.atom_types.numpy(), data_batch.lengths.numpy().reshape(
            -1), \
            data_batch.angles.numpy().reshape(-1), data_batch.frac_coords.numpy()
        gt_atom_types_list.append(atom_types)
        gt_lengths_list.append(lengths)
        gt_angles_list.append(angles)
        gt_frac_coords_list.append(frac_coords)
    gt_structure_list = get_structure(gt_atom_types_list, gt_lengths_list, gt_angles_list, gt_frac_coords_list)

    runner = Runner(conf, score_norm_path)
    runner.model.load_state_dict(torch.load(model_path))

    gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list = runner.generate(num_gen,
                                                                                                   train_data_path)
    gen_structure_list = get_structure(gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list)
    is_valid, validity = MetricsForGeneration.getSMACTValidity(gen_atom_types_list)
    print("SMACTValidity: {}".format(validity))
    is_valid, structure_validity = MetricsForGeneration.getStructureValidity(gen_atom_types_list, gen_lengths_list,
                                                                             gen_angles_list, gen_frac_coords_list,
                                                                             gen_structure_list)
    print("structureValidity: {}".format(structure_validity))

    elemEMD = MetricsForGeneration.getElemTypeNumEMD(gen_atom_types_list, gt_atom_types_list)
    print("element EMD: {}".format(elemEMD))
    densityEMD = MetricsForGeneration.getDensityEMD(gen_structure_list, gt_structure_list)
    print("density EMD: {}".format(densityEMD))

