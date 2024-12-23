# 材料计算评测框架

## 黑盒资源利用率模型评测

### 目标

- 对所有模型，希望评估如下指标：
  - 训练耗时, 内存利用率, CPU占用率, GPU占用率, 模型参数量, FLOPS。
- 对于材料表示学习/材料表征/声子计算领域的属性预测任务，希望评估如下指标：
  - 回归任务：MAE, MSE, RMSE
  - 分类任务：ROC, AUC, Accuracy, Precision, Recall, F1_Score
- 对于材料生成任务，希望评估如下指标：
  - 有效率
  - EMD

 

### 支持的模型

- 目前希望支持材料表示学习/材料表征/声子计算/材料生成领域的部分模型：

  - SyMat, Matformer, CrystalMELA_ExRT

  

### Demo结果

| Metrics/Model                         | **SyMat**                | Matformer | CrystalMELA_ExRT |
| ------------------------------------- |--------------------------| ---- | ---- |
| **duration (s)**                      | 59.766264567151666       | 279.6747203390005 | 6.047079102998396 |
| **gpu:0/gpu_utilization (%)/mean**    | 44.72449783248215        | 57.34024483103515 | 11.003273684208766 |
| **gpu:0/memory_utilization (%)/mean** | 20.88040779519485        | 52.77920022097266 | 6.495060309199788 |
| **gpu:0/power_usage (W)/mean**        | 59.759263886971425       | 137.61544282401448 | 17.624458588602128 |
| **host/cpu_percent (%)/mean**         | 11.093333485733483       | 10.778991993015806 | 60.34570476629634 |
| **host/memory_percent (%)/mean**      | 61.39133474256576        | 57.97314795422511 | 67.23119746163277 |
| **Params** | 144 M (trainset) | 191 M (trainset) | - (ExRT是传统机器学习模型) |
| **FLOPs** | 565 G (trainset) | 4507 G (trainset) | - |
| **MAE** | - | 0.0625682767954642 | - |
| **MSE** | - | 0.00973441919158266 | - |
| **RMSE** | - | 0.09866316025539959 | - |
| **有效率** | 0.9/1.0(numGen=10) | - | - |
| **EMD** | 0.258/1.075(numGen=10) | - | - |
| **ROC/AUC** | - | -                   | 0.18400880762056013/0.03733669441730265 |
| **Accuracy** | - | - | 0.918989320135452/0.802031779109143 |
| **Precision** | - | - | 0.8820606355813501/0.6898393250597135 |
| **Recall** | - | - | 0.8518414383176423/0.5316257366414497 |
| **F1_Score** | - | - | 0.86343144768573/0.6269558037116416 |

### 黑盒资源利用率

- SyMat模型 & perov_5数据集 & 50epoch的资源利用率采样情况

![](assets/SyMat_BlackBoxResource_show.png)


- CrystalMELA_ExRT
![](assets/CrystalMELA_ExRT_BlackBoxResource_show.png)


- Matformer模型 & 200epoch
![](assets/Matformer_200Epoch_BlackBoxResource_show.png)


### 运行

~~~bash
# [SyMat, Matformer, CrystalMELA_ExRT]
python modelMain.py --modelName SyMat
~~~



### 需要用到的包：

~~~bash
### nvitop/thop/psutil
# nvitop
pip3 install --upgrade nvitop   # conda install -c conda-forge nvitop
# thop
pip install thop

# 除此之外，你还需要安装对应模型运行所需的包。如 Models/{modelName}/README 所示。
# SyMat for SyMat && CrystalMELA_ExRT
# Matformer for Matformer
~~~





## 黑盒资源利用率框架评测

### 目标

- 支持材料学经典模型的Torch/Paddle/MindSpore实现，并分析不同框架实现的highLevel区别
- 如应用的整体资源利用率


### 支持的模型

- CGCNN, NequIP, VGNN



### Demo结果

<img src="assets/CGCNN_show2.png" style="zoom: 67%;" />

<img src="assets/NequIP_show2.png" style="zoom: 67%;" />

### 运行

~~~bash
# [CGCNN, NequIP, VGNN]
python frameMain.py --modelName NequIP
~~~



### 需要用到的包：

~~~bash
### nvitop
# nvitop
pip3 install --upgrade nvitop   # conda install -c conda-forge nvitop

# 除此之外，你还需要安装对应模型运行所需的包。如 Framework/{modelName}/README 所示。
# CGCNN_paddle for CGCNN_paddle
# SyMat for CGCNN_torch/NequIP_torch
# NequIP_mindspore for NequIP_mindspore
# VGNN_torch for VGNN_torch
# VGNN_mindspore for VGNN_mindspore
~~~



## 白盒插桩方法框架评测

### 目标

- 支持材料学共性和个性评测用例的Torch/Paddle/MindSpore实现，并分析不同框架实现的lowLevel区别
- 如某算子实现的计算、访存的时延和吞吐


### 支持的用例

- GCN, E3NN

### Nsys_Demo结果
- 详见: [`CodeInstrumentation_README`](./CodeInstrumentation/README.md)


### NCU_Demo结果
- 指标细节详见：[`CodeInstrumentation_metrics_needto_collect`](./CodeInstrumentation/info/metrics_needto_collect.txt)

- GCN 卷积算子实现的对比 torch & paddle
  <img src="assets/GCN_convlayer_ncu_metrics_show.png" />

  | ConvLayer | Torch | Paddle |
  | --------- | ----- | ------ |
  | 耗时(ms)  | 0.341 | 0.482  |

#### E3NN 各种算子实现的对比 torch & mindspore
- E3NN_FullyConnectedNet
<img src="assets/E3NN_FullyConnectNet_ncu_metrics_prof.png" />

- E3NN_Gate
<img src="assets/E3NN_Gate_ncu_metrics_prof.png" />

- E3NN_Scatter
<img src="assets/E3NN_Scatter_ncu_metrics_prof.png" />

- E3NN_TensorProduct
<img src="assets/E3NN_TensorProduct_ncu_metrics_prof.png" />

### 运行
~~~bash
# 见 CodeInstrumentation/TestCase_*/profile.sh
~~~


### 需要用到的包：
~~~bash
### nvtx
pip install nvtx -i https://pypi.douban.com/simple/

# 除此之外，你还需要安装对应用例运行所需的包。
# CGCNN_paddle for GCN_paddle
# SyMat for GCN_torch
~~~


# jax
~~~bash
conda create -n jax python=3.8
~~~

