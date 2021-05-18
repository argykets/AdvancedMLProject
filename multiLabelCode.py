from scipy.io import loadmat
import pandas as pd
import random
import numpy as np
import ActiveLearning
import Plots as plots
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn import metrics

# to avoid future warnings for sklearn
import warnings

from BinaryRelevanceClassifier import BinaryRelevanceClassifier
from BinaryRelevanceClassifierUS import BinaryRelevanceClassifierUS
from ClassifierChains import ClassifierChains

warnings.filterwarnings("ignore")


def preprocessing():
    dataset = loadmat('CHD_49/CHD_49.mat')

    X = dataset['data']
    X = pd.DataFrame(X)

    y = dataset['targets']
    y = pd.DataFrame(y, columns=['class1', 'class2', 'class3', 'class4', 'class5', 'class6'])

    # Convert target values into 0 and 1 representation
    y = y.replace(-1, 0)

    print('Descriptive stats: ')
    features = y.columns
    print(features)
    print(y[features].sum())
    # print('X shape:', X.shape)
    # print('y shape:', y.shape)

    # Normalize data
    X = (X - X.min()) / (X.max() - X.min())

    # Split to training and test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)

    return x_train, x_test, y_train, y_test


def gridSearchClassifier(classifier):
    cv_folds = 5

    # Set up the parameter grid to search
    param_grid = {'base_classifier': [DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=2),
                                      RandomForestClassifier(criterion='entropy'),
                                      LogisticRegression(), GaussianNB(), KNeighborsClassifier(), SVC()]}

    # Perform the search
    # Using the custom accuracy function defined earlier
    tuned_model = GridSearchCV(classifier, param_grid, scoring=make_scorer(accuracy_score),
                               verbose=2, n_jobs=-1, cv=cv_folds)
    tuned_model.fit(X_train, y_train)

    # Print details of the best model
    print("Best Parameters Found: ")
    print(tuned_model.best_params_)
    print(tuned_model.best_score_)


def accuracy_score(y_test, y_pred):
    # y_pred is a numpy array, y_test is a dataframe
    # to compare the two, convert to a single type
    y_test = y_test.to_numpy()

    # shape of test and preds must be equal
    assert y_test.shape == y_pred.shape
    i = 0
    # list of scores for each training sample
    scores = []

    # for each test sample
    while i < len(y_test):
        count = 0
        # count the number of matches in the sample
        # y_test[i] -> row values in test set (true values)
        # y_pred[i] -> row values in predictions set (predicted values)
        for p, q in zip(y_test[i], y_pred[i]):
            if p == q:
                count += 1

        # accuracy score for the sample = no. of correctly predicted labels/total no. of labels
        scores.append(count / y_pred.shape[1])
        i += 1

        # final accuracy = avg. accuracy over all test samples =
    # sum of the accuracy of all training samples/no. of training samples
    return round((sum(scores) / len(y_test)), 5)


def gridSearchClassifierChains():
    cv_folds = 5

    # generate 20 random orders for class labels
    rand_orders = [list(range(1, y_test.shape[1] + 1)) for i in list(range(1, 20))]

    for lst in rand_orders:
        random.shuffle(lst)

    # make sure natural order is present
    rand_orders.append([1, 2, 3, 4, 5, 6])

    # Set up the parameter grid to search
    param_grid = {'base_classifier': [DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=2),
                                      RandomForestClassifier(criterion='entropy'),
                                      LogisticRegression, GaussianNB(), KNeighborsClassifier(), SVC()],
                  'order': rand_orders}

    # Perform the search
    tuned_model = GridSearchCV(ClassifierChains(),
                               param_grid, scoring=make_scorer(accuracy_score), verbose=2, n_jobs=-1, cv=cv_folds)
    tuned_model.fit(X_train, y_train)

    # Print details
    print("Best Parameters Found: ")
    print(tuned_model.best_params_)
    print(tuned_model.best_score_)


if __name__ == "__main__":
    # Import CHD dataset and split to training and test set
    X_train, X_test, y_train, y_test = preprocessing()
    # print("X_train.shape: " + str(X_train.shape))
    # print("X_test.shape: " + str(X_test.shape))
    # print("y_train.shape: " + str(y_train.shape))
    # print("y_test.shape: " + str(y_test.shape))

    # Instantiate the classifiers
    # gridSearchClassifier(BinaryRelevanceClassifier())
    br_clf = BinaryRelevanceClassifier(LogisticRegression())
    # gridSearchClassifier(BinaryRelevanceClassifierUS())
    brus_clf = BinaryRelevanceClassifierUS(RandomForestClassifier(criterion='entropy'))
    # gridSearchClassifierChains()
    cc = ClassifierChains(SVC(), order=[6, 4, 3, 2, 5, 1])

    # Fit
    br_clf.fit(X_train, y_train)
    brus_clf.fit(X_train, y_train)
    cc.fit(X_train, y_train)

    # Predict
    y_pred = br_clf.predict(X_test)
    y_pred_us = brus_clf.predict(X_test)
    y_pred_cc = cc.predict(X_test)

    print('Accuracy score using BR algo: ', (accuracy_score(y_test, y_pred)))
    print('Accuracy score using BR with US algo: ', (accuracy_score(y_test, y_pred_us)))
    print('Accuracy score using Classifier Chains algo: ', (accuracy_score(y_test, y_pred_cc)))

    # Compare all methods with all base classifiers
    # list of base models
    base_models = [DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=2),
                   RandomForestClassifier(criterion='entropy'),
                   LogisticRegression(), GaussianNB(), KNeighborsClassifier(), SVC()]
    base_model_names = ["Decision Tree", "Random Forest", "Logistic Regression", "GaussianNB", "kNN", "SVM"]

    # store accuracy scores
    br_clf_accuracies = dict()
    br_clfus_accuracies = dict()

    # store F1 scores
    br_clf_f1 = dict()
    br_clfus_f1 = dict()

    i = 0
    for clf in base_models:
        # without undersampling
        br_clf = BinaryRelevanceClassifier(clf)
        br_clf.fit(X_train, y_train)
        br_y_pred = br_clf.predict(X_test)

        # find accuracy using custom accuracy function defined
        accuracy = accuracy_score(y_test, br_y_pred)
        br_clf_accuracies[base_model_names[i]] = accuracy

        # find f1 score using sklearn
        y_pred_df = pd.DataFrame(br_y_pred)
        f1_score_br = metrics.f1_score(y_test, y_pred_df, average='macro')
        br_clf_f1[base_model_names[i]] = f1_score_br

        # with undersampling
        brus_clf = BinaryRelevanceClassifierUS(clf)
        brus_clf.fit(X_train, y_train)
        brus_y_pred = brus_clf.predict(X_test)

        # find accuracy using custom accuracy function defined
        accuracy_us = accuracy_score(y_test, brus_y_pred)
        br_clfus_accuracies[base_model_names[i]] = accuracy_us

        # find f1 score using sklearn
        y_pred_df = pd.DataFrame(brus_y_pred)
        f1_score_us = metrics.f1_score(y_test, y_pred_df, average='macro')
        br_clfus_f1[base_model_names[i]] = f1_score_us

        i += 1

    print("===================Accuracy Scores=====================")
    print("Binary Relevance")
    print(br_clf_accuracies)
    print("Binary Relevance with Under-Sampling")
    print(br_clfus_accuracies)

    print("======================F1 Scores========================")
    print("Binary Relevance")
    print(br_clf_f1)
    print("Binary Relevance with Under-Sampling")
    print(br_clfus_f1)

    # plots.plotUnderSampling(br_clf_accuracies,br_clfus_accuracies,br_clf_f1,br_clfus_f1)

    # list of base models
    base_models = [DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=2),
                   RandomForestClassifier(criterion='entropy'),
                   LogisticRegression(max_iter=20000), GaussianNB(), KNeighborsClassifier(), SVC()]
    base_model_names = ["Decision Tree", "Random Forest", "Logistic Regression", "GaussianNB", "kNN", "SVM"]

    cc_accuracies = dict()
    cc_f1 = dict()

    i = 0
    for clf in base_models:
        cc = ClassifierChains(clf, order=[6, 4, 3, 2, 5, 1])
        cc.fit(X_train, y_train)
        cc_pred = cc.predict(X_test)
        # accuracy score
        accuracy = accuracy_score(y_test, cc_pred)
        cc_accuracies[base_model_names[i]] = accuracy
        # F1 score
        cc_f1_score = metrics.f1_score(y_test, pd.DataFrame(cc_pred), average='macro')
        cc_f1[base_model_names[i]] = cc_f1_score
        i += 1

    print("====================Classifier Chains Accuracy====================")
    print(cc_accuracies)
    print("===================Classifier Chains F1 Scores====================")
    print(cc_f1)
    # plots.plotClassifierChains(br_clf_accuracies, br_clfus_accuracies, cc_accuracies, br_clf_f1, br_clfus_f1, cc_f1)
    # x_train, x_holdout, y_train, y_holdout = train_test_split(X_train, y_train.values,
    #                                                           random_state=0, test_size=0.9)
    x_train, x_holdout, y_train, y_holdout = train_test_split(np.vstack(X_train.values), np.vstack(y_train.values),
                                                              random_state=0, test_size=0.9)
    ActiveLearning.ActiveLearning(x_train, y_train, x_holdout, y_holdout, X_test, y_test,1)
    ActiveLearning.ActiveLearning(x_train, y_train, x_holdout, y_holdout, X_test, y_test,2)
    ActiveLearning.ActiveLearning(x_train, y_train, x_holdout, y_holdout, X_test, y_test,3)
