from datetime import datetime
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB


def multilabel_evaluation(y_pred, y_test):
    '''
    Micro accuracy, recall, precision, f1_score evaluation
    '''
    from sklearn.metrics import hamming_loss, multilabel_confusion_matrix
    multilabel_cm = multilabel_confusion_matrix(y_pred, y_test)

    tn = multilabel_cm[:, 0, 0]
    tp = multilabel_cm[:, 1, 1]
    fp = multilabel_cm[:, 0, 1]
    fn = multilabel_cm[:, 1, 0]
    ac, p, r = [], [], []
    for i in range(len(tp)):
        ac.append((tp[i] + tn[i]) / (tn[i] + tp[i] + fn[i] + fp[i]))
        p.append(0 if tp[i] == 0 and fp[i] == 0 else tp[i] / (tp[i] + fp[i]))
        r.append(0 if tp[i] == 0 and fn[i] == 0 else tp[i] / (tp[i] + fn[i]))

    accuracy = np.around(np.mean(ac), 3)
    precision = np.around(np.mean(p), 3)
    recall = np.around(np.mean(r), 3)
    f1_score = np.around(2 * recall * precision / (recall + precision), 3)
    hamming = np.around(hamming_loss(y_test, y_pred), 3)
    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'hamming_loss': hamming}


def plotActiveLearner(accuracy_res_U, precision_res_U, recall_res_U, f1_res_U, hamming_loss_res_U,query_strategy):
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(accuracy_res_U)
    ax.plot(precision_res_U)
    ax.plot(recall_res_U)
    ax.plot(f1_res_U)
    ax.plot(hamming_loss_res_U)
    ax.scatter(range(len(accuracy_res_U)), accuracy_res_U, s=10)
    ax.scatter(range(len(precision_res_U)), precision_res_U, s=10)
    ax.scatter(range(len(recall_res_U)), recall_res_U, s=10)
    ax.scatter(range(len(f1_res_U)), f1_res_U, s=10)
    ax.scatter(range(len(hamming_loss_res_U)), hamming_loss_res_U, s=10)
    ax.legend(['Accuracy', 'Percision', 'Recall', 'F1-micro', 'Hamming'])

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.set_title(query_strategy)
    ax.set_xlabel('Query Iteration')
    ax.set_ylabel('Classification Performance')

    plt.show()
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]

def ActiveLearningRandom(x_train_actual, y_train_actual, x_holdout, y_holdout, x_test, y_test, n):
    learner = ActiveLearner(
        estimator=OneVsRestClassifier(BernoulliNB(class_prior=None, alpha=.7)),
        query_strategy=random_sampling,
        X_training=x_train_actual, y_training=y_train_actual)
    #
    y_pred = learner.predict(x_test)
    starting_res = multilabel_evaluation(y_pred, y_test)
    f1_res = [starting_res["f1_score"]]
    N_QUERIES = 315
    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(x_holdout)

        # Teach our ActiveLearner model the record it has requested.
        X, y = x_holdout[query_index].reshape(1, -1), y_holdout[query_index].reshape(1, -1)
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        x_holdout, y_holdout = np.delete(x_holdout, query_index, axis=0), np.delete(y_holdout, query_index, axis=0)

        if (index % 25) == 0:
            prediction = learner.predict(x_test)
            results = multilabel_evaluation(prediction, y_test)
            f1_res.append(results["f1_score"])

    prediction = learner.predict(x_test)
    results = multilabel_evaluation(prediction, y_test)
    print(results)
    f1_res.append(results["f1_score"])

    return f1_res
def ActiveLearning(x_train_actual, y_train_actual, x_holdout, y_holdout, x_test, y_test, n):
    print("")
    print("Starting Active Learning...")
    print("")
    start = datetime.now()
    query_strategy = entropy_sampling
    if n == 1:
        query_strategy = margin_sampling
    elif n == 2:
        query_strategy = uncertainty_sampling

    learner = ActiveLearner(
        estimator=OneVsRestClassifier(BernoulliNB(class_prior=None, alpha=.7)),
        query_strategy=query_strategy,
        X_training=x_train_actual, y_training=y_train_actual)
    #
    y_pred = learner.predict(x_test)
    starting_res = multilabel_evaluation(y_pred, y_test)
    accuracy_res = [starting_res["accuracy"]]
    precision_res = [starting_res["precision"]]
    recall_res = [starting_res["recall"]]
    f1_res = [starting_res["f1_score"]]
    hamming_loss_res = [starting_res["hamming_loss"]]
    print(x_train_actual.shape)
    print(y_train_actual.shape)

    xreturn = np.empty((1, 49))
    yreturn = np.empty((1, 6))
    N_QUERIES = 315
    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(x_holdout)

        # Teach our ActiveLearner model the record it has requested.
        X, y = x_holdout[query_index[0]].reshape(1, -1), y_holdout[query_index[0]].reshape(1, -1)
        learner.teach(X=X, y=y)
        xreturn = np.concatenate((xreturn, X))
        yreturn =  np.concatenate((yreturn, y))
        # Remove the queried instance from the unlabeled pool.
        x_holdout, y_holdout = np.delete(x_holdout, query_index, axis=0), np.delete(y_holdout, query_index, axis=0)

        if (index % 25) == 0:
            prediction = learner.predict(x_test)
            results = multilabel_evaluation(prediction, y_test)

            accuracy_res.append(results["accuracy"])
            precision_res.append(results["precision"])
            recall_res.append(results["recall"])
            f1_res.append(results["f1_score"])
            hamming_loss_res.append(results["hamming_loss"])

    prediction = learner.predict(x_test)
    results = multilabel_evaluation(prediction, y_test)
    print(results)
    accuracy_res.append(results["accuracy"])
    precision_res.append(results["precision"])
    recall_res.append(results["recall"])
    f1_res.append(results["f1_score"])
    hamming_loss_res.append(results["hamming_loss"])
    query_strategy = "Entropy sampling"
    if n == 1:
        query_strategy = "Margin sampling"
    elif n == 2:
        query_strategy = "Uncertainty sampling"
    #plotActiveLearner(accuracy_res, precision_res, recall_res, f1_res, hamming_loss_res,query_strategy)
    print('Finished active learning in : ', datetime.now() - start)
    return np.delete(xreturn,obj=0, axis=0),np.delete(yreturn,obj=0, axis=0),f1_res
