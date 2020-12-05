from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from package.util.func import *
import warnings
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings('ignore')
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier

def model_RF(features, num_class=4, over_sampling=True):
    X = features
    # Extract the labels for training
    labels = X['target']

    # Remove the target
    X = X.drop(columns=['target'])

    if over_sampling:
        print("Raw target type ratio:")
        plot_fraction(labels, num_class=num_class)
        sample_solver = SMOTETomek(random_state=326)
        X, labels = sample_solver.fit_sample(X, labels)
        print("After imbalance processing type ratio:")
        plot_fraction(labels, num_class=num_class)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels, test_size=0.3, random_state=326)

    # Extract the ids
    test_ids = X_test['ID']

    # Remove the ids and target
    X_train = X_train.drop(columns=['ID'])
    X_test = X_test.drop(columns=['ID'])
    # Extract feature names
    feature_names = list(X_train.columns)

    print('Training Data Shape: ', X_train.shape)
    print('Testing Data Shape: ', X_test.shape)

    model = BalancedRandomForestClassifier(n_estimators=100, random_state=326)
    model.fit(X_train, y_train)

    test_predictions = model.predict_proba(X_test)
    plot_auc(y_test, test_predictions, num_class= num_class)

    test_predictions = get_result(test_predictions)
    print("RF balance score is ", balanced_accuracy_score(y_test, test_predictions))
    test_predict = pd.DataFrame({'ID': test_ids, 'predict_class': test_predictions})
    test_real = pd.DataFrame({'ID': test_ids, 'real_class': y_test})

    return test_real, test_predict
