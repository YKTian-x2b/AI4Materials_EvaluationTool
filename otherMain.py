import numpy as np
import csv
import os

from Utils.draw import readCSV, readCSV_v2, draw, drawForFrame

current_dir = os.path.dirname(os.path.abspath(__file__))


def readV1(filePath):
    readCSV(filePath)


def readV2(filePath):
    readCSV_v2(filePath)
    draw(filePath)


if __name__ == '__main__':
    configForMode = ["Models", "Framework"]
    configForMetrics = [["Matformer", "matformer/matformer_mp_bulk"],
                        ["CrystalMELA_ExRT", "res"],
                        ["SyMat", "tmpRes"],
                        ["CGCNN", "res/TorchRes"],
                        ["CGCNN", "res/PaddleRes"],
                        ["E3NN/NequIP", "res/TorchRes"],
                        ["E3NN/NequIP", "res/MindsporeRes"],
                        ]

    # Idx = 4
    # filePath_ = os.path.join(current_dir, configForMode[1],
    #                          configForMetrics[Idx][0],
    #                          configForMetrics[Idx][1],
    #                          "metrics.csv")
    # # readV1(filePath_)
    # readV2(filePath_)

    idx1 = 5
    filePath_1 = os.path.join(current_dir, configForMode[1],
                              configForMetrics[idx1][0],
                              configForMetrics[idx1][1],
                              "metrics.csv")
    idx2 = 6
    filePath_2 = os.path.join(current_dir, configForMode[1],
                              configForMetrics[idx2][0],
                              configForMetrics[idx2][1],
                              "metrics.csv")
    savePath = os.path.join(current_dir, "Framework/E3NN/NequIP/")
    drawForFrame(filePath_1, filePath_2, "torch", "mindspore", savePath)
