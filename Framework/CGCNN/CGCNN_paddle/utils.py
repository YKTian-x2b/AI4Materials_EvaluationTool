import paddle
import numpy as np
from sklearn import metrics
import shutil
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

class Normalizer(object):
    """
    Normalize a Tensor and restore it later.
    """
    def __init__(self, tensor):
        self.mean = paddle.mean(tensor)
        self.std = paddle.to_tensor(np.std(tensor.numpy()), dtype='float32')

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


def mae(prediction, target):
    return paddle.mean(paddle.abs(target - prediction))


def save_checkpoint(state, is_best, filename=current_dir+'checkpoint.pth.tar'):
    paddle.save(state, filename)
    if is_best:
        shutil.copyfile(filename, current_dir+'model_best.pth.tar')
