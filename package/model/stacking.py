# -*- coding:utf-8 -*-
from package.preprocess.load_data import read_data
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from package.util.func import *
from imblearn.ensemble import BalancedRandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

def stacking_model(features,num_class = 4, RANDOM_SEED=326, test_size=0.3):
    labels = features['target']
    features = features.drop(columns=['target'])

    print("Raw target type ratio:")
    plot_fraction(labels, num_class=num_class)
    sample_solver = SMOTETomek()
    features, labels = sample_solver.fit_sample(features, labels)
    print("After imbalance processing type ratio:")
    plot_fraction(labels, num_class=num_class)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=test_size, random_state=RANDOM_SEED)

    # Extract the ids
    test_ids = X_test['ID']

    # Remove the ids and target
    X_train = X_train.drop(columns=['ID'])
    X_test = X_test.drop(columns=['ID'])
    # Extract feature names
    feature_names = list(X_train.columns)

    print('Training Data Shape: ', X_train.shape)
    print('Testing Data Shape: ', X_test.shape)

    # Convert to np arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    clf1 = BalancedRandomForestClassifier(n_estimators=100, random_state=326)


    clf2 = xgb.XGBClassifier(
            # 初始得分
            base_score=0.5,
            # 基分类器类型
            booster='gbtree',
            # 树最大深度，越大越容易过拟合
            max_depth=11,
            # 学习率
            learning_rate=0.03,
            # 估计器的数量
            n_estimators=1700,
            # 目标参数
            objective='multi:softmax',
            # 线程数
            n_jobs=-1,
            # 分裂所需的最小损失减少量
            gamma=0.03,
            # 拆分节点权重和的阈值
            min_child_weight=1,
            # 最大权重估计，1-10，数字越大越保守
            max_delta_step=0,
            # 随机选取的比例
            subsample=0.5,
            # 每棵树特征选取范围
            colsample_bytree=0.8,
            # L1正则化参数
            reg_alpha=1,
            # L2正则化参数
            reg_lambda=1,
            seed=0,
            missing=None, num_class=num_class)


    lr = LogisticRegression()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=lr,
                                random_state=RANDOM_SEED)
    sclf.fit(X_train, y_train)
    test_predictions = sclf.predict_proba(X_test)
    plot_auc(y_test, test_predictions, num_class=4)

    test_predictions = get_result(test_predictions)
    test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
    test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
    print("balanced_acc_score is ", balanced_accuracy_score(y_test, test_predictions))

    return test_real, test_predict



