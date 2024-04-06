import os
import sys
import argparse

from Models.SyMat import runSyMat
# from Models.Matformer.matformer import runMatformer


def runModel(modelName):
    if modelName == 'SyMat':
        print('SyMat')
        runSyMat.run()
    elif modelName == 'Matformer':
        print('Matformer')
        # runMatformer.run()
    elif modelName == 'PDos':
        print('PDos')
    else:
        raise ValueError("The model should be one of [SyMat, Matformer, PDos]")


def parse_args():
    parser = argparse.ArgumentParser(description='AI for Material Evaluation Tool')
    parser.add_argument('--modelName', type=str, required=True, help='需要输入模型名称！')
    # parser.add_argument('--dataset', type=str, required=True, help='需要输入数据集名称或文件位置！')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runModel(args.modelName)
    # python main.py --modelName Matformer
