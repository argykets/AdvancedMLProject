#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, plot_roc_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import EasyEnsembleClassifier

from sklearn import svm, datasets
from sklearn.cluster import KMeans

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.under_sampling import CondensedNearestNeighbour

from sklearn.metrics import roc_curve
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import TomekLinks

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
# import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score




def get_categorical_features(df):
    types = df.columns.to_series().groupby(df.dtypes).groups
    dict_of_types = {k.name: v for k, v in types.items()}
    objects_list = dict_of_types.get('object').tolist()
    return objects_list


def one_hot_encoding(df, col_name):
    one_hot = pd.get_dummies(df[col_name])
    df = df.drop(col_name, axis=1)
    df = df.join(one_hot)
    return df


def model_evaluation(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report_imbalanced(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f' % accuracy)
    print('Balanced accuracy: %.2f' % balanced_acc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print("fpr: ", fpr)
    print("tpr: ", tpr)
    print(thresholds)

    plot_precision_recall_curve(clf, X_test, y_test, pos_label=0)
    plot_roc_curve(clf, X_test, y_test)
    plt.show()



    #     print('Recall: %.2f' % recall_score(y_test, y_pred))
    #     print('Precision: %.2f' % precision_score(y_test, y_pred))
    #     print('F1: %.2f' % f1_score(y_test, y_pred))

def scatter2D(x, y):
    x['stroke'] = y
    df = x
    fig, ax = plt.subplots(figsize=(15, 8))

    colors = {0: 'blue', 1: 'red'}
    sizevalues = {0: 5, 1: 20}
    alphavalues = {0: 0.4, 1: 0.8}
    ax.scatter(df['age'], df['avg_glucose_level'],
               c=df['stroke'].apply(lambda x: colors[x]),
               s=df['stroke'].apply(lambda x: sizevalues[x]),
               alpha= .5)

    plt.show()


def scatter_plot(x, y):

    # normalization
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    # 2-dimensional PCA
    pca = PCA(n_components=3)
    pca.fit(x)
    x = pd.DataFrame(pca.transform(x))
    print(pca.explained_variance_ratio_)

    # φτιαξε αυτο για να μη χαλαει το x_train πριν την εκπαιδευση του RF
    x['stroke'] = y
    pcaDF = x
    pcaDF.columns = ['pc1', 'pc2', 'pc3', 'stroke']

    # 3D plot
    fig = plt.figure()
    ax = Axes3D(fig)

    # positive class
    positive = pcaDF[pcaDF['stroke'] == 1]
    ppx = positive['pc1']
    ppy = positive['pc2']
    ppz = positive['pc3']

    # negative class
    negative = pcaDF[pcaDF['stroke'] == 0]
    pnx = negative['pc1']
    pny = negative['pc2']
    pnz = negative['pc3']

    # plotting
    ax.scatter(pnx, pny, pnz, label='Class 2', c='blue')
    ax.scatter(ppx, ppy, ppz, label='Class 1', c='red')
    
    plt.show()


def compute_accuracy(TP, FP, TN, FN):
    print()


# aggregate predictions of each fold
def aggregate_labels(labels_list):
    y = []
    for l in labels_list:
        y.extend(l)
    return y
    
    
def perf_measure(y_actual, y_pred):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i] == y_pred[i] == 0:
            TP += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 1:
            TN += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN


def evaluate_model(X, y, oversampling):
    skf = StratifiedKFold(10, shuffle=True, random_state=42)

#     predicted_targets = np.array([])
#     actual_targets = np.array([])

#     tp_list = []
#     fp_list = []
#     tn_list = []
#     fn_list = []

    y_test_agg_list = []
    y_pred_agg_list = []
    for train_index, test_index in skf.split(X, y):
#             print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
            pos_df = y_test[y_test["stroke"] == 1]
            neg_df = y_test[y_test["stroke"] == 0]

            print('positive cases: ', len(pos_df))
            print('negative cases: ', len(neg_df))
        
        
            if oversampling:            
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
                ros = RandomOverSampler()
                X_train, y_train = ros.fit_resample(X_train, kmeans.labels_)


            # Fit the classifier
            classifier = svm.SVC().fit(X_train, y_train)
#             classifier = RandomForestClassifier().fit(X_train, y_train)

            # Predict the labels of the test set samples
            y_pred = classifier.predict(X_test)
            print(y_pred)
            
            print(type(y_test))
            print(type(y_pred))
            
            for yt in y_test.values.ravel():
                y_test_agg_list.append(yt)
            
            for yp in y_pred.tolist():
                y_pred_agg_list.append(yp)
            
            
#             y_test_agg_list.append(y_test.values.ravel())
#             y_pred_agg_list.append(y_pred.tolist())
            
#             tp, fp, tn, fn = perf_measure(y_test.values.ravel(), y_pred.tolist())
            
#             tp_list.append(tp)
#             fp_list.append(fp)
#             tn_list.append(tn)
#             fn_list.append(fn)

#             predicted_targets = np.append(predicted_targets, predicted_labels)
#             actual_targets = np.append(actual_targets, y_test)
    
#     tp_avg = sum(tp_list) / len(tp_list)
#     fp_avg = sum(fp_list) / len(fp_list)
#     tn_avg = sum(tn_list) / len(tn_list)
#     fn_avg = sum(fn_list) / len(fn_list)
    
#     y_test = aggregate_labels(y_test_agg_list)
#     print(y_test)
#     y_pred = aggregate_labels(y_pred_agg_list)
#     print(y_pred)
    
    return y_pred_agg_list, y_test_agg_list
    
#     return tp_avg, fp_avg, tn_avg, fn_avg


def kNNUndersampling(X_train, X_test, y_train, y_test):
    # define the undersampling method
    print('UNDERSAMPLING: ')
    undersample = CondensedNearestNeighbour(n_neighbors=1)
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def brSMOTE(X_train, X_test, y_train, y_test):
    # borderline SMOTE
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
    sm = BorderlineSMOTE()
    # sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(y_train['stroke'].value_counts())
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)

    # remove TomekLinks
    tl = TomekLinks(sampling_strategy='auto')
    X_train, y_train = tl.fit_resample(X_train, y_train)
    print(y_train['stroke'].value_counts())
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def adaptiveSynthetic(X_train, X_test, y_train, y_test):
    ada = ADASYN()
    X_train, y_train = ada.fit_resample(X_train, y_train)
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def kMeansRos(X_train, X_test, y_train, y_test):
    kmeans = KMeans(n_clusters=2).fit(X_train)
    y_train = kmeans.labels_

    print(kmeans.cluster_centers_)

    countPos = np.count_nonzero(y_train == 0)
    countNeg = np.count_nonzero(y_train == 1)


    if countPos < countNeg:
        indices_one = y_train == 1
        indices_zero = y_train == 0
        y_train[indices_one] = 0  # replacing 1s with 0s
        y_train[indices_zero] = 1  # replacing 0s with 1s

    scatter_plot(X_train, y_train)


    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    model_evaluation(X_train, X_test, y_train, y_test)

if __name__ == '__main__':

    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print(df.describe())

    print("Total number of entries: ", len(df))

    # positive/negative
    pos_df = df[df["stroke"] == 1]
    neg_df = df[df["stroke"] == 0]

    # print length
    print('positive cases: ', len(pos_df))
    print('positive cases: ', len(neg_df))

    # Drop id column & drop Missing values
    df = df.drop(['id'], axis=1)
    print(df.isnull().sum())
    df = df.dropna()
    print("After removal of missing values: ", len(df))

    # Separate features and ground truth label
    # X = df.drop(['stroke'], axis=1)
    # y = df[['stroke']]

    categorical_features = get_categorical_features(df)
    print('categorical features: ', categorical_features)

    # one hot encoding for categorical features
    for feature in categorical_features:
        df = one_hot_encoding(df, feature)


    print(df.columns)

    print(df['stroke'].value_counts())

    corrDF = df.drop(["stroke"], axis=1).apply(lambda x: x.corr(df.stroke.astype('category').cat.codes))
    print(type(corrDF))
    print(corrDF.abs().sort_values(ascending=False))

    scoring = ['accuracy', 'balanced_accuracy']

    X = df.drop(['stroke'], axis=1)
    y = df[['stroke']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

    from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print(y_train['stroke'].value_counts())
    # print(y_test['stroke'].value_counts())
    print('NO PREPROCESSING: ')
    scatter_plot(X_train, y_train)
    # scatter2D(X, y)
    print(y_train['stroke'].value_counts())
    model_evaluation(X_train, X_test, y_train, y_test)


    brSMOTE(X_train, X_test, y_train, y_test)
    # X_train['stroke'] = kmeans.labels_

    # k-fold cross-validation
    # y_pred, y_test = evaluate_model(X, y, True)

    # model_evaluation(y_pred, y_test)


    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
    # print(y_test['stroke'].unique())

    # ros = RandomOverSampler()
    # X_ros, y_ros = ros.fit_resample(X_train, kmeans.labels_)

    # print(X_ros)

    # scatter_plot(X_ros, y_ros)
    # print(y_ros['stroke'].value_counts())
    # print(Counter(y_ros))
#
    # from sklearn.metrics import plot_confusion_matrix

    # plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    # plt.show()



