# 材料计算模型评估框架

### 目标

- 对所有模型，希望评估如下指标：
  - 训练耗时, 内存利用率, CPU占用率, GPU占用率, 模型参数量, FLOPS。

- 对于材料表示学习/材料表征/声子计算领域的属性预测任务，希望评估如下指标：
  - 回归任务：MAE, MSE, RMSE
  - 分类任务：ROC, AUC, Accuracy, Precision, Recall, F1_Score

- 对于材料生成任务，希望评估如下指标：
  - 有效率：
  - 成功率：
  - COV
  - EMD

 

### 支持的模型

- 目前希望支持材料表示学习/材料表征/声子计算/材料生成领域的部分模型：

  - FTCP, SyMat, PDos, Matformer

  

### Demo结果

| Metrics/Model                         | **SyMat**          | FTCP | PDos | Matformer | Other |
| ------------------------------------- | ------------------ | ---- | ---- | --------- | ----- |
| **duration (s)**                      | 59.766264567151666 |      |      |           |       |
| **gpu:0/gpu_utilization (%)/mean**    | 44.72449783248215  |      |      |           |       |
| **gpu:0/memory_utilization (%)/mean** | 20.88040779519485  |      |      |           |       |
| **gpu:0/power_usage (W)/mean**        | 59.759263886971425 |      |      |           |       |
| **host/cpu_percent (%)/mean**         | 11.093333485733483 |      |      |           |       |
| **host/memory_percent (%)/mean**      | 61.39133474256576  |      |      |           |       |



### 运行

~~~bash
python main.py --modelName SyMat
~~~



### 需要用到的包：

~~~bash
### nvitop/thop/psutil
# nvitop
pip3 install --upgrade nvitop
conda install -c conda-forge nvitop
# thop
pip install thop
~~~