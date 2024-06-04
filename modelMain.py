import os
import sys
import argparse


def runModel(modelName):
    if modelName == 'SyMat':
        from Models.SyMat import runSyMat
        print('SyMat')
        runSyMat.run()
    elif modelName == 'Matformer':
        from Models.Matformer.matformer import runMatformer
        print('Matformer')
        runMatformer.run()
    elif modelName == 'PDos':
        print('PDos')
    elif modelName == 'CrystalMELA_ExRT':
        from Models.CrystalMELA_ExRT import runCrystalMELA_ExRT
        print('CrystalMELA_ExRT')
        runCrystalMELA_ExRT.run()
    else:
        raise ValueError("The model should be one of [SyMat, Matformer, PDos, CrystalMELA_ExRT]")


def parse_args():
    parser = argparse.ArgumentParser(description='AI for Material Evaluation Tool')
    parser.add_argument('--modelName', type=str, required=True, help='需要输入模型名称！')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runModel(args.modelName)
    # python modelMain.py --modelName Matformer
    # python modelMain.py --modelName CrystalMELA_ExRT
