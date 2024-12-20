# 1.修改目的地址
# 2.在本地命令行执行 就是另起一个命令行

# local to server
scp -P 3434 ./*.json  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci\
/AI4Materials_EvaluationTool/CodeInstrumentation/TestCase_E3NN/data/phonon

# server to local
scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci\
/AI4Materials_EvaluationTool/CodeInstrumentation/TestCase_GCN/profile_713.ncu-rep /home/yujixuan/Downloads


################ for GCN
scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_GCN/res/PaddleRes/ncu_metrics_show_paddle.txt   ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_GCN/res/PaddleRes/ncu_metrics_prof_paddle.txt   ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_GCN/res/TorchRes/ncu_metrics_show_torch.txt   ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_GCN/res/TorchRes/ncu_metrics_prof_torch.txt   ./


scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_GCN/res/PaddleRes/convLayer_time_cost.txt   ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_GCN/res/TorchRes/convLayer_time_cost.txt   ./


###################### for E3NN
scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_E3NN/res/MindsporeRes/ncu_metrics_prof_mindspore_FullyConnectedNet.txt ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_E3NN/res/TorchRes/ncu_metrics_prof_torch_FullyConnectedNet.txt ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_E3NN/res/TorchRes/ncu_metrics_prof_torch_Gate.txt ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_E3NN/res/MindsporeRes/ncu_metrics_prof_mindspore_Gate.txt ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_E3NN/res/MindsporeRes/ncu_metrics_prof_mindspore_Scatter.txt ./

scp -P 3434  yujixuan@10.134.138.222:/home/yujixuan/AI4Sci/AI4Materials_EvaluationTool\
/CodeInstrumentation/TestCase_E3NN/res/TorchRes/ncu_metrics_prof_torch_Scatter.txt ./