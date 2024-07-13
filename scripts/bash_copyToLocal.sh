# 1.修改目的地址
# 2.在本地命令行执行 就是另起一个命令行

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