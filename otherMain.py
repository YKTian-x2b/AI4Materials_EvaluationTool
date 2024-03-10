from Utils.utils import readCSV
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    filePath = os.path.join(current_dir, "Models", "SyMat", "metrics.csv")
    readCSV(filePath)