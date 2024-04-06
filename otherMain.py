from Utils.utils import readCSV, readCSV_v2, draw
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # innerPath = 'matformer/matformer_mp_bulk'
    # filePath = os.path.join(current_dir, "Models", "Matformer",
    #                         innerPath, "metrics.csv")
    # readCSV(filePath)

    innerPath = 'tmpRes'
    filePath = os.path.join(current_dir, "Models", "SyMat",
                            innerPath, "metrics.csv")
    draw(filePath)
    # readCSV_v2(filePath)
