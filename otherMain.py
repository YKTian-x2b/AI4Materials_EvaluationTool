from Utils.utils import readCSV, readCSV_v2, draw
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


def readV1(filePath):
    readCSV(filePath)


def readV2(filePath):
    readCSV_v2(filePath)
    draw(filePath)


if __name__ == '__main__':
    configForMetrics = [["Matformer", "matformer/matformer_mp_bulk"],
                        ["CrystalMELA_ExRT", "res"],
                        ["SyMat", "tmpRes"]]
    Idx = 1
    filePath_ = os.path.join(current_dir, "Models",
                             configForMetrics[Idx][0],
                             configForMetrics[Idx][1],
                             "metrics.csv")
    readV1(filePath_)
    # readV2
