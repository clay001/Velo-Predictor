# -*- coding:utf-8 -*-
from sklearn.linear_model import LogisticRegressionCV
from package.util.func import *
from imblearn.over_sampling import SMOTE,  ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss, ClusterCentroids, RepeatedEditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection, NeighbourhoodCleaningRule, RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier, \
EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
import gc

def plot_auc_temp(y_test, y_score, num_class, name, c='r'):
    # not show
    y_one_hot = label_binarize(y_test, classes=[i for i in range(num_class)])
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.ravel())
    # micro_auc = auc(fpr, tpr)
    macro_auc = roc_auc_score(y_one_hot, y_score, average="macro")
    plt.plot(fpr, tpr, c=c, lw=1, alpha=0.7, label= name + '=%.3f' % macro_auc)


def origin( X, num_class=4):
    # Extract the labels for training
    labels = X['target']
    target_names = [str(i) for i in range(num_class)]

    # Remove the target
    X = X.drop(columns=['target'])

    print("Raw target type ratio:")
    plot_fraction(labels, num_class=num_class)
    plt.figure(dpi=600)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels, test_size=0.3,
                                                            random_state=326)
    X_train = X_train.drop(columns=["ID"])
    # Extract the ids
    test_ids = X_test['ID']
    X_test = X_test.drop(columns=['ID'])

    # Extract feature names
    feature_names = list(X_train.columns)

    model = RandomForestClassifier(random_state=0)

    # 原始是黑色
    model.fit(X_train, y_train)
    test_predictions = model.predict_proba(X_test)
    plot_auc_temp(y_test, test_predictions, num_class, name="origin", c="r")
    test_predictions = get_result(test_predictions)
    print(" balance score is ", balanced_accuracy_score(y_test, test_predictions))

    test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
    test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
    print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                    target_names=target_names))

    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=8)
    plt.title('ROC and AUC of Origin methods', fontsize=17)
    plt.show()
    return


def test_oversample(X, num_class=4):
    # Extract the labels for training
    labels = X['target']
    target_names = [str(i) for i in range(num_class)]

    # Remove the target
    X = X.drop(columns=['target'])

    print("Raw target type ratio:")
    plot_fraction(labels, num_class = num_class)
    plt.figure(dpi=600)
    colorlist = ["b", "g", "r", "c", "m", "y", "k"]
    modelist = ["SMOTE", "ADASYN", "BLS", "SVM"]

    for i in range(len(modelist)):
        model_name = modelist[i]
        if model_name == "SMOTE":
            sample_solver = SMOTE()
        elif model_name == "ADASYN":
            sample_solver = ADASYN()
        elif model_name == "BLS":
            sample_solver = BorderlineSMOTE()
        elif model_name == "SVM":
            sample_solver = SVMSMOTE()
        X_all, y_all = sample_solver.fit_sample(X, labels)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all, test_size=0.3, random_state=326)

        X_train = X_train.drop(columns=["ID"])
        # Extract the ids
        test_ids = X_test['ID']
        X_test = X_test.drop(columns=['ID'])

        # Extract feature names
        feature_names = list(X_train.columns)

        model = RandomForestClassifier(random_state=0)

        # 原始是黑色
        model.fit(X_train, y_train)
        test_predictions = model.predict_proba(X_test)
        plot_auc_temp(y_test, test_predictions, num_class, name=model_name, c=colorlist[i])
        test_predictions = get_result(test_predictions)
        print(model_name+ " balance score is ", balanced_accuracy_score(y_test, test_predictions))

        test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
        test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
        print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                    target_names=target_names))
        gc.enable()
        del model
        gc.collect()

    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=8)
    plt.title('ROC and AUC of Oversampling methods', fontsize=17)
    plt.show()
    return

def test_undersample(X, num_class=4):
    # Extract the labels for training
    labels = X['target']
    target_names = [str(i) for i in range(num_class)]

    # Remove the target
    X = X.drop(columns=['target'])

    print("Raw target type ratio:")
    plot_fraction(labels, num_class=num_class)
    plt.figure(dpi=600)
    colorlist = ["b", "g", "r", "c", "m", "y", "k"]
    modelist = ["NearMiss", "CC", "RUS", "RENN", \
                "NCR", "OSS"]

    for i in range(len(modelist)):
        model_name = modelist[i]
        if model_name == "NearMiss":
            sample_solver = NearMiss()
        elif model_name == "CC":
            sample_solver = ClusterCentroids(random_state=0)
        elif model_name == "RUS":
            sample_solver = RandomUnderSampler(random_state=0)
        elif model_name == "RENN":
            sample_solver = RepeatedEditedNearestNeighbours()
        elif model_name == "NCR":
            sample_solver = NeighbourhoodCleaningRule()
        elif model_name == "OSS":
            sample_solver = OneSidedSelection(random_state=0)
        X_all, y_all = sample_solver.fit_sample(X, labels)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all, test_size=0.3,
                                                            random_state=326)

        X_train = X_train.drop(columns=["ID"])
        # Extract the ids
        test_ids = X_test['ID']
        X_test = X_test.drop(columns=['ID'])

        # Extract feature names
        feature_names = list(X_train.columns)

        model = RandomForestClassifier(random_state=0)

        # 原始是黑色
        model.fit(X_train, y_train)
        test_predictions = model.predict_proba(X_test)
        plot_auc_temp(y_test, test_predictions, num_class, name=model_name, c=colorlist[i])
        test_predictions = get_result(test_predictions)
        print(model_name+ " balance score is ", balanced_accuracy_score(y_test, test_predictions))

        test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
        test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
        print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                    target_names=target_names))
        gc.enable()
        del model
        gc.collect()

    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=8)
    plt.title('ROC and AUC of Undersampling methods', fontsize=17)
    plt.show()
    return

def test_combine(X, num_class=4):
    # Extract the labels for training
    labels = X['target']
    target_names = [str(i) for i in range(num_class)]

    # Remove the target
    X = X.drop(columns=['target'])

    print("Raw target type ratio:")
    plot_fraction(labels, num_class=num_class)
    plt.figure(dpi=600)
    colorlist = ["b", "g", "r", "c", "m", "y", "k"]
    modelist = ["SMOTETomek", "SMOTEENN"]

    for i in range(len(modelist)):
        model_name = modelist[i]
        if model_name == "SMOTETomek":
            sample_solver = SMOTETomek()
        elif model_name == "SMOTEENN":
            sample_solver = SMOTEENN(random_state=0)
        X_all, y_all = sample_solver.fit_sample(X, labels)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all, test_size=0.3,
                                                            random_state=326)

        X_train = X_train.drop(columns=["ID"])
        # Extract the ids
        test_ids = X_test['ID']
        X_test = X_test.drop(columns=['ID'])

        # Extract feature names
        feature_names = list(X_train.columns)

        model = RandomForestClassifier(random_state=0)

        # 原始是黑色
        model.fit(X_train, y_train)
        test_predictions = model.predict_proba(X_test)
        plot_auc_temp(y_test, test_predictions, num_class, name=model_name, c=colorlist[i])
        test_predictions = get_result(test_predictions)
        print(model_name+ " balance score is ", balanced_accuracy_score(y_test, test_predictions))

        test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
        test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
        print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                    target_names=target_names))
        gc.enable()
        del model
        gc.collect()

    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=8)
    plt.title('ROC and AUC of Combine methods', fontsize=17)
    plt.show()
    return

def test_ensemble(X, num_class=4):
    # Extract the labels for training
    labels = X['target']
    target_names = [str(i) for i in range(num_class)]

    # Remove the target
    X = X.drop(columns=['target'])

    print("Raw target type ratio:")
    plot_fraction(labels, num_class=num_class)
    plt.figure(dpi=600)
    colorlist = ["b", "g", "r", "c", "m", "y", "k"]
    modelist = ["LogisticCV", "RandomForest", "BBC", "RUSBoostC",\
                "EasyEC", "BRF"]

    for i in range(len(modelist)):
        model_name = modelist[i]
        # X_all, y_all = sample_solver.fit_sample(X, labels)

        X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels, test_size=0.3,
                                                            random_state=326)

        X_train = X_train.drop(columns=["ID"])
        # Extract the ids
        test_ids = X_test['ID']
        X_test = X_test.drop(columns=['ID'])

        # Extract feature names
        feature_names = list(X_train.columns)

        if model_name == "LogisticCV":
            model = LogisticRegressionCV(class_weight='balanced', solver='lbfgs', multi_class="multinomial")
        elif model_name == "RandomForest":
            model = RandomForestClassifier(random_state=0)
        elif model_name == "BBC":
            model = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                              sampling_strategy='auto',
                                              replacement=False,
                                              random_state=0)
        elif model_name == "BRF":
            model = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
        elif model_name == "RUSBoostC":
            model = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R', random_state=0)
        elif model_name == "EasyEC":
            model = EasyEnsembleClassifier(random_state=0)

        model.fit(X_train, y_train)
        test_predictions = model.predict_proba(X_test)
        plot_auc_temp(y_test, test_predictions, num_class, name=model_name, c=colorlist[i])
        test_predictions = get_result(test_predictions)
        print(model_name+ " balance score is ", balanced_accuracy_score(y_test, test_predictions))

        test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})
        test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
        print(classification_report(list(test_real["real_class"]), list(test_predict["predict_class"]),
                                    target_names=target_names))
        gc.enable()
        del model
        gc.collect()

    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=8)
    plt.title('ROC and AUC of ensemble methods', fontsize=17)
    plt.show()
    return


if __name__ == "__main__":
    from package.preprocess.load_data import read_data
    X = read_data(dataset="dentategyrus", use_gene_list=True)
    origin(X)
    #test_oversample(X)
    #test_undersample(X)
    #test_ensemble(X)