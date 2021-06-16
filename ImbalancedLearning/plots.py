import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotBars(y1, y2, y3):
    # importing package
    import matplotlib.pyplot as plt
    import numpy as np

    # create data
    x = np.arange(4)
    width = 0.2

    # plot data in grouped manner of bar type
    plt.bar(x - 0.2, y1, width, color='midnightblue', tick_label=y1)
    plt.bar(x, y2, width, color='steelblue')
    plt.bar(x + 0.2, y3, width, color='lightblue')
    plt.xticks(x, ["Original", "RUS", "Borderline SMOTE", "ADASYN"])
    plt.ylabel("Scores")
    plt.legend(['Accuracy', 'Balanced_Accuracy', 'AUC'])
    plt.title('Methods Performance Comparison')

    plt.show()


if __name__ == '__main__':

    resultsDF = pd.read_csv('results.csv')
    print(resultsDF.head())
    plotBars(resultsDF['Accuracy'].tolist(), resultsDF['Balanced_Accuracy'].tolist(), resultsDF['AUC'].tolist())