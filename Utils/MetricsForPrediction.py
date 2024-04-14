import numpy as np
from sklearn import metrics


def getMSE(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)


def getRMSE(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def getMAE(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)


# (TP+TN)/(TP+TN+FP+FN) 预测正确的结果占总样本的百分比
def getAccuracy(y_true, y_pred, normalize=True):
    return metrics.accuracy_score(y_true, y_pred, normalize=normalize)


# TP/(TP+FN) 在实际为正的样本中被预测为正样本的概率
def getRecall(y_true, y_pred, average='binary', zero_division=np.nan):
    return metrics.recall_score(y_true, y_pred, average=average, zero_division=zero_division)


# TP/(TP+FP) 在所有被预测为正的样本中实际为正的样本的概率
def getPrecision(y_true, y_pred, average='binary', zero_division=np.nan):
    return metrics.precision_score(y_true, y_pred, average=average, zero_division=zero_division)


# 查准率和查全率的平衡考量 2*查准率*查全率 / (查准率+查全率)
def getF1Score(y_true, y_pred, average='binary', zero_division=np.nan):
    return metrics.f1_score(y_true, y_pred, average=average, zero_division=zero_division)


# ROC曲线下面积
def getROCandAUC(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
    return metrics.auc(fpr, tpr)
