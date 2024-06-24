import os
import shutil

source_path = "/opt/AI4Sci/AI4S_EvaluationToolset/AI4Material_EvaluationTool/Framework/CGCNN/data/root_dir"
target_path = "/opt/AI4Sci/AI4S_EvaluationToolset/AI4Material_EvaluationTool/CodeInstrumentation/TestCase_GCN/root_dir"

# 确保目标路径存在
if not os.path.exists(target_path):
    print("path error")
    exit()

# 列表中的数字前缀
prefixes = ["27164.cif",
            "30481.cif",
            "38394.cif",
            "7591.cif",
            "13270.cif",
            "19968.cif",
            "19210.cif",
            "30383.cif"]

# 遍历源路径中的文件
for filename in os.listdir(source_path):
    # 检查文件是否以列表中的某个数字为前缀且后缀为.cif
    if any(filename == prefix for prefix in prefixes):
        source_file = os.path.join(source_path, filename)
        target_file = os.path.join(target_path, filename)

        # 复制文件
        shutil.copy2(source_file, target_file)
