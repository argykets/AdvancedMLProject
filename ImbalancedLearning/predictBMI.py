import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning
import xgboost
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVR

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print(df.describe())

    initDF = df[['id', 'age', 'gender', 'bmi']]
    print('Initial DataFrame size:', len(initDF))

    trainDF = initDF.dropna()
    print('Total cases w/o missing values', len(trainDF))

    nanDF = initDF[initDF.bmi.isna()]
    print('Missing Values cases:', len(nanDF))

    one_hot = pd.get_dummies(trainDF['gender'])
    trainDF = trainDF.drop(['gender'], axis=1)
    trainDF = trainDF.join(one_hot)
    print(trainDF)

    X = trainDF.drop(['id', 'bmi'], axis=1)
    y = trainDF[['bmi']]

    print(X)
    print(y)



    scaler = MinMaxScaler()
    clf = RandomForestRegressor()
    # clf = XGBRegressor()
    # clf = SVR(kernel='sigmoid')
    pipeline = Pipeline(steps=[('scaler', scaler), ('clf', clf)])

    # evaluate the pipeline
    folds = KFold(n_splits=10, shuffle=True)
    n_scores = cross_validate(pipeline, X, y, scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'], cv=folds, n_jobs=-1)
    print('MAE: %.3f (%.3f)' % (np.mean(n_scores['test_neg_mean_absolute_error']), np.std(n_scores['test_neg_mean_absolute_error'])))
    print('RMSE: %.3f (%.3f)' % (np.mean(n_scores['test_neg_root_mean_squared_error']), np.std(n_scores['test_neg_root_mean_squared_error'])))

    print(nanDF)

    pipeline.fit(X, y)
    print(X)

    X_test = nanDF.drop(['id', 'bmi'], axis=1)
    one_hot = pd.get_dummies(X_test['gender'])
    X_test = X_test.drop(['gender'], axis=1)
    X_test = X_test.join(one_hot)
    print(X_test)
    X_test['Other'] = 0

    y_pred = pipeline.predict(X_test)
    print(y_pred)

    nanDF['bmi'] = y_pred

    print(nanDF)

    # nanDF.to_csv('BMI_approx.csv')
