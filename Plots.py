import matplotlib.pyplot as plt
import matplotlib as mpl

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

def randomSampling(f1Active,f1Random):
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(f1Active)
    ax.plot(f1Random)

    ax.scatter(range(len(f1Active)), f1Active, s=10)
    ax.scatter(range(len(f1Random)), f1Random, s=10)

    ax.legend(['Uncertainty-Sampling', 'Random-Sampling'])

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.set_title('F1')
    ax.set_xlabel('Query Iteration')
    ax.set_ylabel('Classification Performance')

    plt.show()
def plotActiverOverAll(f1,f1_2,f1_3):
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(f1)
    ax.plot(f1_2)
    ax.plot(f1_3)

    ax.scatter(range(len(f1)), f1, s=10)
    ax.scatter(range(len(f1_2)), f1_2, s=10)
    ax.scatter(range(len(f1_3)), f1_3, s=10)

    ax.legend(['Margin-Sampling', 'Uncertainty-Sampling', 'Entropy-Sampling'])

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.set_title('F1')
    ax.set_xlabel('Query Iteration')
    ax.set_ylabel('Classification Performance')

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
