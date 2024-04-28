import os
import sys
import argparse

from Framework.CGCNN import run_CGCNN


def runModel(modelName):
    if modelName == 'CGCNN':
        print('CGCNN')
        run_CGCNN.run()
    elif modelName == 'E3NN':
        print('E3NN')
        # runMatformer.run()
    else:
        raise ValueError("The model should be one of [CGCNN, E3NN]")


def parse_args():
    parser = argparse.ArgumentParser(description='AI for Material Evaluation Tool')
    parser.add_argument('--modelName', type=str, required=True, help='需要输入模型名称！')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runModel(args.modelName)
    # python frameMain.py --modelName CGCNN
