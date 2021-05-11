import pandas as pd
from scipy.io import loadmat
import pandas as pd
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
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
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    # Normalize data
    X = (X - X.min()) / (X.max() - X.min())

    # Split to training and test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)

    return x_train, x_test, y_train, y_test


class BinaryRelevanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=LogisticRegression()):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        """Build a Binary Relevance classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples, n_labels]
            The target values (class labels) as integers or strings.
        """

        # list of individual classifiers
        self.models = []

        # for every class label
        for label in list(y.columns):
            # Check that X and y have correct shape
            x_checked, y_checked = check_X_y(X, y[label])
            # every classifier is independent of the others
            # hence we create a copy of the base classifier instance
            base_model = clone(self.base_classifier)
            # fit the base model - one model each for Y1, Y2....Y14
            basel_model = base_model.fit(x_checked, y_checked)
            # add the fitted model list of individual classifiers
            self.models.append(base_model)

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])
        X = check_array(X)

        all_preds = pd.DataFrame()
        i = 0
        # list of individual classifier predictions
        preds = []

        # predict against each fitted model - one model per label
        for model in self.models:
            pred = model.predict(X)
            # add the prediction to the dataframe
            preds.append(pd.DataFrame({'Class' + str(i + 1): pred}))
            i += 1

        # dataframe with predictions for all class labels
        all_preds = pd.concat(preds, axis=1)
        # standard sklearn classifiers return predictions as numpy arrays
        # hence convert the dataframe to a numpy array
        return all_preds.to_numpy()

    def predict_proba(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])
        X = check_array(X)

        all_preds = pd.DataFrame()
        i = 0

        for model in self.models:
            # Call predict_proba of the each base model
            pred = model.predict_proba(X)
            # Add the probabilities of 1 to the dataframe
            all_preds['Class' + str(i + 1)] = [one_prob[1] for one_prob in pred]
            i += 1

        # return probabilities
        return all_preds.to_numpy()


class BinaryRelevanceClassifierUS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=LogisticRegression(max_iter=20000)):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        """Build a Binary Relevance classifier with Under sampling from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples]
            The target values (class labels) as integers or strings.
        """

        # list of individual classifiers
        self.models = []

        # for each class label
        for label in list(y.columns):

            X_cp = X.copy()
            # pick the column values for the label
            y_cp = y[label]

            # sampling is done on both X and y, hence join the two dataframes
            X_y_data = pd.concat([X_cp, y_cp], axis=1)

            # counters for 0 values and 1 values
            n_val0, n_val1 = 0, 0

            j = 0
            # for each sample
            while j < len(X_y_data):
                # if value for the label is 0
                if (X_y_data.iloc[j][label] == 0):
                    n_val0 += 1
                else:
                    # value 1
                    n_val1 += 1
                j += 1

            # under sample the majority class
            # randomly pick samples from majority class equal to the number of samples in the minority class
            # both the classes will have the same number of samples
            if n_val0 > n_val1:
                # majority 0 values
                val1 = X_y_data[X_y_data[label] == 1]
                val0 = X_y_data[X_y_data[label] == 0].sample(n_val1)

                X_y_data = pd.concat([val0, val1], axis=0)

            elif n_val1 > n_val0:
                # majority 1 values
                val1 = X_y_data[X_y_data[label] == 1].sample(n_val0)
                val0 = X_y_data[X_y_data[label] == 0]

                X_y_data = pd.concat([val0, val1], axis=0)

            # split back into X and y
            X_cp = X_y_data.iloc[:, :-1]
            y_cp = X_y_data.iloc[:, -1]

            base_model = clone(self.base_classifier)
            # fit the base model - one model each for Y1, Y2....Y14
            a, b = check_X_y(X_cp, y_cp)
            base_model.fit(a, b)
            # list of individual classifiers classifiers
            self.models.append(base_model)

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])
        X = check_array(X)

        all_preds = pd.DataFrame()
        i = 0
        # list of individual classifier predictions
        preds = []

        # for every fitted model
        for model in self.models:
            # predict for X
            pred = model.predict(X)
            # add to the list of predictions
            preds.append(pd.DataFrame({'Class' + str(i + 1): pred}))
            i += 1

        # store predictions for each label in a single dataframe
        all_preds = pd.concat(preds, axis=1)
        # standard sklearn classifiers return predictions as numpy arrays
        # hence convert the dataframe to a numpy array
        return all_preds.to_numpy()

    def predict_proba(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])
        X = check_array(X)

        all_preds = pd.DataFrame()
        i = 0

        for model in self.models:
            # Call predict_proba of the each base model
            pred = model.predict_proba(X)
            # Add the probabilities of 1 to the dataframe
            all_preds['Class' + str(i + 1)] = [one_prob[1] for one_prob in pred]
            i += 1

        # return probabilities
        return all_preds.to_numpy()


class ClassifierChains(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=LogisticRegression(max_iter=20000), order=None):
        self.base_classifier = base_classifier
        self.order = order

    def fit(self, X, y):
        """
        Build a Classifier Chain from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples, n_labels]
            The target values (class labels) as integers or strings.

        """

        # check the order parameter
        if self.order is None:
            # default value - natural order for number of labels
            self.order = list(range(y.shape[1]))
        elif self.order == 'random':
            # random order
            self.order = list(range(y.shape[1]))
            random.shuffle(self.order)
        else:
            # order specified
            if (len(self.order) == y.shape[1]):
                # expect order from 1, hence reduce 1 to consider zero indexing
                self.order = [o - 1 for o in self.order]

        # list of base models for each class
        self.models = [clone(self.base_classifier) for clf in range(y.shape[1])]

        # create a copy of X
        X_joined = X.copy()
        # X_joined.reset_index(drop=True, inplace=True)

        # create a new dataframe with X and y-in the order specified
        # if order = [2,4,5,6...] -> X_joined= X, y2, y4, y5...
        for val in self.order:
            X_joined = pd.concat([X_joined, y['class' + str(val + 1)]], axis=1)

        # for each ith model, fit the model on X + y0 to yi-1 (in the order specified)
        # if order = [2,4,6,....] fit 1st model on X for y2, fit second model on X+y2 for y4...
        for chain_index, model in enumerate(self.models):
            # select values of the class in order
            y_vals = y.loc[:, 'class' + str(self.order[chain_index] + 1)]
            # pick values for training - X+y upto the current label
            t_X = X_joined.iloc[:, :(X.shape[1] + chain_index)]
            check_X_y(t_X, y_vals)
            # fit the model
            model.fit(t_X, y_vals)

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):

        # check if the models list has been set up
        check_is_fitted(self, ['models'])

        # dataframe to maintain previous predictions
        pred_chain = pd.DataFrame(columns=['class' + str(o + 1) for o in self.order])

        X_copy = X.copy()
        X_joined = X.copy()

        # use default indexing
        X_joined.reset_index(drop=True, inplace=True)
        X_copy.reset_index(drop=True, inplace=True)

        i = 0

        # for each ith model, predict based on X + predictions of all models upto i-1
        # happens in the specified order since models are already fitted according to the order
        for chain_index, model in enumerate(self.models):
            # select previous predictions - all columns upto the current index
            prev_preds = pred_chain.iloc[:, :chain_index]
            # join the previous predictions with X
            X_joined = pd.concat([X_copy, prev_preds], axis=1)
            # predict on the base model
            pred = model.predict(X_joined)
            # add the new prediction to the pred chain
            pred_chain['class' + str(self.order[i] + 1)] = pred
            i += 1

        # re-arrange the columns in natural order to return the predictions
        pred_chain = pred_chain.loc[:, ['class' + str(j + 1) for j in range(0, len(self.order))]]
        # all sklearn implementations return numpy array
        # hence convert the dataframe to numpy array
        return pred_chain.to_numpy()

    # Function to predict probabilities of 1s
    def predict_proba(self, X):
        # check if the models list has been set up
        check_is_fitted(self, ['models'])

        # dataframe to maintain previous predictions
        pred_chain = pd.DataFrame(columns=['class' + str(o + 1) for o in self.order])
        # dataframe to maintain probabilities of class labels
        pred_probs = pd.DataFrame(columns=['class' + str(o + 1) for o in self.order])
        X_copy = X.copy()
        X_joined = X.copy()

        # use default indexing
        X_joined.reset_index(drop=True, inplace=True)
        X_copy.reset_index(drop=True, inplace=True)

        i = 0

        # for each ith model, predict based on X + predictions of all models upto i-1
        # happens in the specified order since models are already fitted according to the order
        for chain_index, model in enumerate(self.models):
            # select previous predictions - all columns upto the current index
            prev_preds = pred_chain.iloc[:, :chain_index]
            # join the previous predictions with X
            X_joined = pd.concat([X_copy, prev_preds], axis=1)
            # predict on the base model
            pred = model.predict(X_joined)
            # predict probabilities
            pred_proba = model.predict_proba(X_joined)
            # add the new prediction to the pred chain
            pred_chain['class' + str(self.order[i] + 1)] = pred
            # save the probabilities of 1 according to label order
            pred_probs['class' + str(self.order[i] + 1)] = [one_prob[1] for one_prob in pred_proba]
            i += 1

        # re-arrange the columns in natural order to return the probabilities
        pred_probs = pred_probs.loc[:, ['class' + str(j + 1) for j in range(0, len(self.order))]]
        # all sklearn implementations return numpy array
        # hence convert the dataframe to numpy array
        return pred_probs.to_numpy()


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
    print("X_train.shape: " + str(X_train.shape))
    print("X_test.shape: " + str(X_test.shape))
    print("y_train.shape: " + str(y_train.shape))
    print("y_test.shape: " + str(y_test.shape))

    # Instantiate the classifiers
    #gridSearchClassifier(BinaryRelevanceClassifier())
    br_clf = BinaryRelevanceClassifier(LogisticRegression())
    #gridSearchClassifier(BinaryRelevanceClassifierUS())
    brus_clf = BinaryRelevanceClassifierUS(RandomForestClassifier(criterion='entropy'))
    #gridSearchClassifierChains()
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

    plt.plot(list(br_clf_accuracies.keys()), list(br_clf_accuracies.values()))
    plt.plot(list(br_clfus_accuracies.keys()), list(br_clfus_accuracies.values()))
    plt.xticks(rotation=90)
    plt.legend(['Binary Relevance', 'Binary Relevance Under-Sampling'])
    plt.show()

    plt.plot(list(br_clf_f1.keys()), list(br_clf_f1.values()))
    plt.plot(list(br_clfus_f1.keys()), list(br_clfus_f1.values()))
    plt.xticks(rotation=90)
    plt.legend(['Binary Relevance', 'Binary Relevance Under-Sampling'])
    plt.show()

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

    plt.plot(list(br_clf_accuracies.keys()), list(br_clf_accuracies.values()))
    plt.plot(list(br_clfus_accuracies.keys()), list(br_clfus_accuracies.values()))
    plt.plot(list(cc_accuracies.keys()), list(cc_accuracies.values()))
    plt.xticks(rotation=90)
    plt.legend(['Binary Relevance', 'BR-Under-Sampling', 'Classifier Chains'])
    plt.show()

    plt.plot(list(br_clf_f1.keys()), list(br_clf_f1.values()))
    plt.plot(list(br_clfus_f1.keys()), list(br_clfus_f1.values()))
    plt.plot(list(cc_f1.keys()), list(cc_f1.values()))
    plt.xticks(rotation=90)
    plt.legend(['Binary Relevance', 'BR-Under-Sampling', 'Classifier Chains'])
    plt.show()
