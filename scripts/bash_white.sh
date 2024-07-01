 #!/bin/bash
set -e

echo '--------------------------------------------'
echo '白盒插桩方法演示'
echo '--------------------------------------------'
echo ''
sleep 4s

echo ''
echo '======================================'
echo 'conda activate CGCNN_paddle'
echo '激活运行时环境'
echo '======================================'
echo ''
sleep 4s
source /home/yujixuan/anaconda3/bin/activate /home/yujixuan/anaconda3/envs/CGCNN_paddle


echo ''
echo '======================================'
echo 'cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/CodeInstrumentation/TestCase_GCN'
echo '进入运行目录'
echo '======================================'
echo ''
sleep 4s
cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/CodeInstrumentation/TestCase_GCN


echo ''
echo '======================================'
echo 'nsys profile --stats=true -f true -o res/gcn_profile_nsys python main.py root_dir > res/gcn_profile_nsys.txt'
echo '运行分析脚本，要十分钟左右'
echo '======================================'
echo ''
sleep 4s

nsys profile --stats=true -f true -o res/gcn_profile_nsys \
python main.py root_dir > res/gcn_profile_nsys.txt

echo ''
echo '======================================'
echo '结果见如上命令行输出'
echo '======================================'
echo ''
sleep 4s