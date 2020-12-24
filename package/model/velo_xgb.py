# -*- coding:utf-8 -*-
# XGB
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from imblearn.combine import SMOTETomek
from package.util.func import *
from sklearn.metrics import balanced_accuracy_score
import gc
import warnings
warnings.filterwarnings('ignore')


# 特征矩阵有ID项，label指示target
def model_regression(features, n_folds=5, use_gene_list=False, test_size=0.2):
    if use_gene_list:
        gene_list = np.load("gene_list.npy").tolist()
        features = pd.DataFrame(features, columns=["ID"] + gene_list + ["target"])

    labels = features['target']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)

    # Extract the ids
    test_ids = X_test['ID']

    # Remove the ids and target
    X_train = X_train.drop(columns=['ID', 'target'])
    X_test = X_test.drop(columns=['ID', 'target'])
    # Extract feature names
    feature_names = list(X_train.columns)

    print('Training Data Shape: ', X_train.shape)
    print('Testing Data Shape: ', X_test.shape)

    # Convert to np arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for train/test predictions
    test_predictions = np.zeros(X_test.shape[0])

    # Lists for recording validation scores
    valid_scores = []

    # Iterate through each fold
    # K折交叉
    for train_indices, valid_indices in k_fold.split(X_train):
        print("##########fold############")
        # Training data for the fold
        train_features, train_labels = X_train[train_indices], y_train[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = X_train[valid_indices], y_train[valid_indices]

        # Create the model
        model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=50, max_depth=9, min_child_weight=6, seed=0,
                                 subsample=0.7, colsample_bytree=0.7, gamma=0.6, reg_alpha=4, reg_lambda=6)
        # Train the model
        # 0是训练集上的表现，1是测试集上的表现
        model.fit(train_features, train_labels,
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)], early_stopping_rounds=5,
                  verbose=10)

        # Record the feature importances
        xgb.plot_importance(model, max_num_features=3, importance_type='gain')
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        # 加权训练结果
        test_predictions += model.predict(X_test) / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold = model.predict(valid_features)

        # Record the best score
        valid_score = my_score(valid_labels, out_of_fold, valid_features)
        valid_scores.append(valid_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # 训练集预测值
    test_predict = pd.DataFrame({'ID': test_ids, 'predict_angle': test_predictions})
    test_real = pd.DataFrame({'ID': test_ids, 'real_angle': y_test})
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    valid_scores.append(np.mean(valid_scores))
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    valid = pd.DataFrame({'fold': fold_names,
                          'valid_score': valid_scores,
                          })
    return test_predict, test_real, feature_importances, valid

@timing
def model_XGBC(features, n_folds=5, num_class=4, plot=True, over_sampling=True, test_size=0.3):
    # Extract the labels
    # classification_transfer(features['target'])
    labels = features['target']
    features = features.drop(columns=['target'])

    if over_sampling:
        print("Raw target type ratio:")
        plot_fraction(labels, num_class=num_class)
        sample_solver = SMOTETomek(random_state=326)
        features, labels = sample_solver.fit_sample(features, labels)
        print("After imbalance processing type ratio:")
        plot_fraction(labels, num_class=num_class)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=test_size)
    # SMOTE
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

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for train/test predictions
    test_predictions = np.zeros((X_test.shape[0], num_class))

    # Lists for recording validation scores
    valid_scores = []

    # Iterate through each fold
    # K折交叉
    fold_num = 0
    for train_indices, valid_indices in k_fold.split(X_train):
        print("=====================fold_" + str(fold_num) + "=====================")
        # Training data for the fold
        train_features, train_labels = X_train[train_indices], y_train[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = X_train[valid_indices], y_train[valid_indices]

        # Create the model
        model = xgb.XGBClassifier(
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

        # Train the model
        # 0是训练集上的表现，1是测试集上的表现
        model.fit(train_features, train_labels,
                  eval_set=[(train_features, train_labels), (valid_features, valid_labels)],
                  early_stopping_rounds=300, eval_metric='mlogloss', verbose=100)

        # Record the feature importances
        # xgb.plot_importance(model, max_num_features=3, importance_type='gain')
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        plot_logloss(model, fold_num)
        # Make predictions
        # 加权训练结果
        test_predictions += model.predict_proba(X_test) / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold = model.predict_proba(valid_features)
        # Record the best score
        valid_score = velo_auc(valid_labels, out_of_fold, num_class)
        valid_scores.append(valid_score)
        fold_num += 1
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    if plot:
        plot_auc(y_test, test_predictions, num_class)
    # 训练集预测值
    test_predictions = get_result(test_predictions)
    test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
    test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
    print("Xgboost balance score is ", balanced_accuracy_score(y_test, test_predictions))

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'features': feature_names, 'importance': feature_importance_values})

    valid_scores.append(np.mean(valid_scores))
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    valid = pd.DataFrame({'fold': fold_names,
                          'valid_score': valid_scores,
                          })
    return test_real, test_predict, feature_importances, valid
