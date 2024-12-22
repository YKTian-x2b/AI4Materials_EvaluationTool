import os
import sys
import argparse
import subprocess


def runModel(frameName, modelName):
    runPathIdx = 0
    if frameName != "torch":
        runPathIdx = 1

    if modelName == 'CGCNN':
        from Framework.CGCNN import run_CGCNN
        print('CGCNN')
        run_CGCNN.run(runPathIdx)
    elif modelName == 'NequIP':
        from Framework.E3NN.NequIP import run_NequIP
        print('NequIP')
        run_NequIP.run(runPathIdx)
    elif modelName == 'VGNN':
        from Framework.E3NN.VGNN import run_VGNN
        print('VGNN')
        run_VGNN.run(runPathIdx)
    else:
        raise ValueError("The model should be one of [CGCNN, NequIP, VGNN]")


def parse_args():
    parser = argparse.ArgumentParser(description='AI for Material Evaluation Tool')

    parser.add_argument('--frameName', type=str, required=True, help='需要输入框架名称！')

    parser.add_argument('--modelName', type=str, required=True, help='需要输入模型名称！')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    frame_name = args.frameName.lower()
    model_name = args.modelName

    assert frame_name in ["torch", "paddle", "mindspore"], "The frame name should be one of [torch, paddle, mindspore]"
    assert model_name in ["CGCNN", "NequIP", "VGNN"], "The model name should be one of [CGCNN, NequIP, VGNN]"

    if frame_name == "paddle":
        assert model_name in ["CGCNN"], "CGCNN model only support frame [torch, paddle]"
    elif frame_name == "mindspore":
        assert model_name in ["NequIP", "VGNN"], "[NequIP, VGNN] model only support frame [torch, mindspore]"

    runModel(frame_name, model_name)


    # python frameMain.py --modelName VGNN --frameName torch
