import os
import itertools
import random

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

random_state = 42


def getDataset():
    # Downloading datasets
    save_dir = "data/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + "train.csv"):
        os.system("wget -O data/test.csv 'https://www.dropbox.com/s/rl0755n11jum3r3/test.csv?dl=1'")
        os.system("wget -O data/train.csv 'https://www.dropbox.com/s/8xjyxub9wavni4v/train.csv?dl=1'")
    # Preparing datasets
    train_data_ = pd.read_csv('data/train.csv')
    test_data_ = pd.read_csv('data/test.csv')
    y_train_c_sys_ = train_data_['crystal_system']
    y_test_c_sys_ = test_data_['crystal_system']
    y_train_sgr_ = train_data_['sg_num']
    y_test_sgr_ = test_data_['sg_num']
    X_train_ = train_data_[[str(i) for i in range(1, 11)] + ['peaknum']]
    X_test_ = test_data_[[str(i) for i in range(1, 11)] + ['peaknum']]
    sgr_nums_ = np.array(list(set(y_train_sgr_)))
    return X_train_, X_test_, y_train_sgr_, y_test_sgr_, y_train_c_sys_, y_test_c_sys_, sgr_nums_, train_data_, test_data_


# To reduce the training time, we set n_estimators=50 which smaller than the article.
# The resulted performance is sligly worse than the original (accuracy 80.46% -> 80.20%).
# If you would like to reproduce the result in the article, please set n_estimators=200.
def getSpaceGroupModel():
    return ExtraTreesClassifier(n_estimators=50, n_jobs=-1, max_depth=25,
                                max_features=None, random_state=random_state)


# To reduce the training time, we set n_estimators=50 which should be 500.
# The resulted performance is  worse than the original (accuracy 92.23% -> 91.89%).
def getCrystalSystemModel():
    return ExtraTreesClassifier(n_estimators=50, max_depth=30,
                                max_features=9, n_jobs=-1,
                                random_state=random_state, warm_start=False)


if __name__ == '__main__':
    classnames = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic']
    X_train, X_test, y_train_sgr, y_test_sgr, \
        y_train_c_sys, y_test_c_sys, sgr_nums, \
        train_data, test_data = getDataset()

    exrt_sgr = getSpaceGroupModel()
    # Training space group classifier
    exrt_sgr.fit(X_train, y_train_sgr)
    # validation
    y_pred_sg = exrt_sgr.predict(X_test)
    print('mean accuracy of space group prediction: ', metrics.accuracy_score(y_test_sgr, y_pred_sg) * 100)

    exrt_crystal_system = getCrystalSystemModel()
    # Training crystal system classifier
    exrt_crystal_system.fit(X_train, y_train_c_sys)
    # validation
    y_pred_system = exrt_crystal_system.predict(X_test)
    print('mean accuracy of crystal system prediction: ', metrics.accuracy_score(y_test_c_sys, y_pred_system) * 100)

    test_pred = pd.DataFrame()
    test_pred['crystal_system_pred'] = exrt_crystal_system.predict(X_test)
    test_pred['sg_num_pred'] = exrt_sgr.predict(X_test)
    test_pred['sg_num_true'] = test_data['sg_num']
    test_pred['crystal_system_true'] = test_data['crystal_system']
    test_pred.to_csv('res/pred_result_test.csv', index=False)