import random

from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import pandas as pd


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
