# -*- coding:utf-8 -*-
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, roc_curve, auc, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from package.preprocess import feature_engineering

# 装饰器，显示函数运行的时间
def timing(func):
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        end = time.time()
        print(str(func) + ' Running time: %s Seconds' % (end - start))
        return r

    return wrapper

def get_gene_list(adata, dataset, top_k=10):
    gene_list = feature_engineering.feature_selection(adata, top_k=top_k)
    np.save("./processed_data/" + dataset + "/gene_list.npy", gene_list)
    return

def get_angle(dx, dy):
    angle = 0
    if dx == 0 and dy > 0:
        angle = 0
    if dx == 0 and dy < 0:
        angle = 180
    if dy == 0 and dx > 0:
        angle = 90
    if dy == 0 and dx < 0:
        angle = 270
    if dx > 0 and dy > 0:
        angle = math.atan(dx / dy) * 180 / math.pi
    elif dx < 0 and dy > 0:
        angle = 360 + math.atan(dx / dy) * 180 / math.pi
    elif dx < 0 and dy < 0:
        angle = 180 + math.atan(dx / dy) * 180 / math.pi
    elif dx > 0 and dy < 0:
        angle = 180 + math.atan(dx / dy) * 180 / math.pi
    return angle


def get_direction(V, num_class):
    component = np.array(V)
    direction = [-1 for i in range(len(component))]
    for i in range(len(component)):
        angle = get_angle(component[i][0], component[i][1])
        direction[i] = angle // (360 / num_class)
    return pd.DataFrame(direction, columns=["target"])


def get_target(V, target_gene, threshold):
    # 小上调，小下调，大上调，大下调
    target = []
    for i in range(len(V[target_gene])):
        if V[target_gene][i] == 0:
            target.append(0)
        # 大上调
        elif V[target_gene][i] >= threshold:
            target.append(1)
        # 大下调
        elif V[target_gene][i] <= -threshold:
            target.append(2)
        # 小上调
        elif 0 < V[target_gene][i] < threshold:
            target.append(3)
        # 小下调
        elif -threshold < V[target_gene][i] < 0:
            target.append(4)

    return pd.DataFrame(np.array(target), columns=["target"])


# model
def classification_transfer(labels):
    y = []
    for i in range(len(labels)):
        if 0 <= labels[i] <= 90:
            y.append(0)
        if 90 <= labels[i] <= 180:
            y.append(1)
        if 180 <= labels[i] <= 270:
            y.append(2)
        if 270 <= labels[i] <= 360:
            y.append(3)
    return y


# R方回归系数
def my_score(y_test, y_predict, X_test):
    n = X_test.shape[0]
    p = X_test.shape[1]
    score = 1 - ((1 - r2_score(y_test, y_predict)) * (n - 1)) / (n - p - 1)
    return score


def velo_auc(y_test, y_score, num_class):
    y_one_hot = label_binarize(y_test, classes=[i for i in range(num_class)])
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.ravel())
    micro_auc = auc(fpr, tpr)
    return micro_auc


def plot_auc(y_test, y_score, num_class):
    y_one_hot = label_binarize(y_test, classes=[i for i in range(num_class)])
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.ravel())
    micro_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % micro_auc)
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title('ROC and AUC', fontsize=17)
    plt.show()


def get_result(y_pred):
    """
    :param y_pred: class probability
    :return: prediction class
    """
    ans = []
    for i in range(y_pred.shape[0]):
        cur = list(y_pred[i])
        loc = cur.index(max(cur))
        ans.append(loc)
    return ans


def plot_fraction_gene(X):
    plt.axes(aspect="equal")
    counts = pd.DataFrame(X)['target'].value_counts()
    plt.pie(x=counts, labels=pd.Series(counts.index).map({0: 'not change', 1: 'up regulate', 2: 'down regulate'}),
            autopct='%.2f%%')
    plt.show()


def plot_fraction(X, num_class):
    plt.axes(aspect="equal")
    counts = pd.DataFrame(X)['target'].value_counts()
    plt.pie(x=counts, labels=pd.Series(counts.index).map({i: str(i) for i in range(num_class)}),
            autopct='%.2f%%')
    plt.show()


def plot_logloss(model, fold_num):
    results = model.evals_result_
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Validate')
    ax.legend()
    plt.xlabel('epochs')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss on fold_' + str(fold_num))
    plt.show()


def plot_fi(feature_importances):
    im = pd.DataFrame({'importance': feature_importances["importance"], 'Features': feature_importances["features"]})
    im = im.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(8, 15))
    sns.barplot(y="Features",
                x="importance",
                data=im.sort_values(by="importance", ascending=False))
    plt.title('Velo_XGBC Features importance(avg over folds)')
    plt.tight_layout()


def eval_result_gene(test_real, test_predict):
    cm = confusion_matrix(np.array(test_real["real_class"]), np.array(test_predict["predict_class"]), labels=[0, 1, 2])
    conf_matrix = pd.DataFrame(cm, index=["not change", "up regulate", "down regulate"],
                               columns=["not change", "up regulate", "down regulate"])
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, cmap="Blues")
    plt.ylabel("True label", fontsize=18)
    plt.xlabel("Predicted label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.savefig("xxx.pdf",bbox_inches="tight")
    plt.show()

    # report table
    target_names = ['not change', 'up regulate', 'down regulate']
    print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                target_names=target_names))
    # kappa
    kappa = cohen_kappa_score(list(test_real["real_class"]), list(test_predict["predict_class"]))
    print("Kappa score: ", kappa)


def eval_result(test_real, test_predict, num_class):
    target_names = [str(i) for i in range(num_class)]
    cm = confusion_matrix(np.array(test_real["real_class"]), np.array(test_predict["predict_class"]),
                          labels=np.arange(num_class))
    conf_matrix = pd.DataFrame(cm, index=target_names,
                               columns=target_names)
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, cmap="Blues")
    plt.ylabel("True label", fontsize=18)
    plt.xlabel("Predicted label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.savefig("xxx.pdf",bbox_inches="tight")
    plt.show()

    # report table
    print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                target_names=target_names))
