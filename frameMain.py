import os
import sys
import argparse


def runModel(modelName):
    if modelName == 'CGCNN':
        from Framework.CGCNN import run_CGCNN
        print('CGCNN')
        run_CGCNN.run()
    elif modelName == 'NequIP':
        from Framework.E3NN.NequIP import run_NequIP
        print('NequIP')
        run_NequIP.run()
    elif modelName == 'VGNN':
        from Framework.E3NN.VGNN import run_VGNN
        print('VGNN')
        run_VGNN.run()
    else:
        raise ValueError("The model should be one of [CGCNN, NequIP, VGNN]")


def parse_args():
    parser = argparse.ArgumentParser(description='AI for Material Evaluation Tool')
    parser.add_argument('--modelName', type=str, required=True, help='需要输入模型名称！')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runModel(args.modelName)
    # python frameMain.py --modelName VGNN
