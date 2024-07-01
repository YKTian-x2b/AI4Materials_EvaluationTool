 #!/bin/bash
set -e

echo '--------------------------------------------'
echo '黑盒资源利用率方法演示'
echo '--------------------------------------------'
echo ''
sleep 4s

echo ''
echo '======================================'
echo 'conda activate Matformer'
echo '激活运行时环境'
echo '======================================'
echo ''
sleep 4s
source /home/yujixuan/anaconda3/bin/activate /home/yujixuan/anaconda3/envs/Matformer


echo ''
echo '======================================'
echo 'cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/ && ls'
echo '进入运行目录'
echo '======================================'
echo ''
sleep 4s
rm -rf Matformer_BlackBoxResource.jpg
cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/ && ls

# 这里告诉评委 没有Matformer_BlackBoxResource.jpg图片
echo ''
echo '======================================'
echo '可以看到当前目录没有Matformer_BlackBoxResource.jpg图片'
echo '======================================'
echo ''
sleep 7s


echo ''
echo '======================================'
echo 'python modelMain.py --modelName Matformer'
echo '运行脚本，训练模型'
echo '======================================'
echo ''
sleep 4s

python modelMain.py --modelName Matformer
# 然后告诉评委 要训练很久 所以我们安排了提前退出

echo ''
echo '======================================'
echo '因为训练要很久，所以我们安排了提前退出'
echo '======================================'
echo ''
sleep 4s


echo ''
echo '======================================'
echo 'cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/Models/Matformer/matformer/matformer_mp_bulk/resList && ls'
echo '进入结果目录'
echo '======================================'
echo ''
sleep 4s
cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/Models/Matformer/matformer/matformer_mp_bulk/resList && ls

echo ''
echo '======================================'
echo '可以看到生成了metrics.csv 和 res_log.txt'
echo '======================================'
echo ''
sleep 4s


echo ''
echo '======================================'
echo 'cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/ && python otherMain.py && ls'
echo '结果可视化'
echo '======================================'
echo ''
sleep 4s

cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/ && python otherMain.py && ls

echo ''
echo '======================================'
echo '可以看到生成了图片Matformer_BlackBoxResource.jpg'
echo '======================================'
echo ''
sleep 4s
