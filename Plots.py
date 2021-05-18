import matplotlib.pyplot as plt


def plotUnderSampling(br_clf_accuracies, br_clfus_accuracies, br_clf_f1, br_clfus_f1):
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


def plotClassifierChains(br_clf_accuracies, br_clfus_accuracies, cc_accuracies, br_clf_f1, br_clfus_f1, cc_f1):
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
