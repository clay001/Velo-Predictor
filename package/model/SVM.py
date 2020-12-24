# -*- coding:utf-8 -*-
import  numpy as np
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
def SVM(X, y):
    # 训练数据
    X = np.array(X)
    # velocity的第一维
    y = np.array(y)
    # 训练集和测试集
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=1)

    # 网格搜索
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
    # 记录训练时间
    t0 = time.time()
    # 训练
    svr.fit(X_tr, y_tr)
    svr_fit = time.time() - t0
    print("SVR用时：", time.time() - t0)

    t0 = time.time()
    # 测试
    y_pre = svr.predict(X_val)
    svr_predict = time.time() - t0
    print("predict用时：", time.time() - t0)
