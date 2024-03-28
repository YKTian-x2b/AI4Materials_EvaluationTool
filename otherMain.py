from Utils.utils import readCSV
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    innerPath = 'matformer/matformer_mp_bulk'
    filePath = os.path.join(current_dir, "Models", "Matformer",
                            innerPath, "metrics.csv")
    readCSV(filePath)
    # total_flops: 4839960784896.0, total_params: 200365128.0