import numpy as np
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from enum import Enum

CLASSIFIER_POOL = {
    'RandomForest': RandomForestClassifier(random_state=0, n_estimators=500),
    'SVM': SVC(kernel='rbf', probability=True, gamma='auto'),
    'KNN': KNeighborsClassifier(),
    'NB': GaussianNB(),
    'DT': tree.DecisionTreeClassifier()
}

class Classifier():
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def get_reward(self, state,  # state: 标记指标是否选取的数组
                   data_path):  # data_path:

        count = len(state)  # 本次选的指标数目
        # print(count)

        data = self.data

        for i in reversed(range(len(state) - 1)):
            if state[i] == 0:
                count -= 1
                for index in range(len(data)):
                    del data[index][i]

        if count == 0:
            return 0
        label = np.array(data)[:, -1]
        data = np.array(data)[:, :-1]

        # precision, recall = classify(data, label, method)
        # return 0
        return self.classify(data, label)

    def classify(self, classifier):
        # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.2, random_state=0)
        # print(y_train)
        # data_test = load_data("data_test.csv")
        # for i in reversed(range(len(state))):
        #     if state[i] == 0:
        #         for index in range(len(data_test)):
        #             del data_test[index][i]
        # y_test = np.array(data_test)[:, -1]
        # x_test = np.array(data_test)[:, :-1]

        classifier.fit(x_train, y_train)

        # y_predict = classifier.predict(x_test)
        # result = metrics.accuracy_score(y_test, y_predict)
        # print(result)
        #
        # data_test = load_data("data_test.csv")
        # state = []
        # for i in range(183):
        #     if i + 1 == 3 or i + 1 == 103 or i + 1 == 168 or i + 1 == 173 or i + 1 == 183:
        #         state.append(1)
        #     else:
        #         state.append(1)
        # for i in reversed(range(len(state))):
        #     if state[i] == 0:
        #         for index in range(len(data_test)):
        #             del data_test[index][i]
        # y_test = np.array(data_test)[:, -1]
        # x_test = np.array(data_test)[:, :-1]
        # print(len(x_test[0]))

        # print(result)
        # return result
        # scores = cross_val_score(classifier, x_train, y_train, cv=10)

        y_predict = classifier.predict(x_test)
        result = metrics.accuracy_score(y_test, y_predict)
        # return result
        #
        # false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test,y_predict)
        # #
        # roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        # print(true_positive_rate)
        # print(false_positive_rate)
        # print(roc_auc)
        return result
        # plt.title('Receiver Operating Characteristic')
        # plt.plot(false_positive_rate, true_positive_rate, 'b',
        #          label='AUC = %0.2f' % roc_auc)
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 0.3])
        # plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
        # return  roc_auc
        # return scores.mean()
        # print(scores.mean())
        # return precision_score(y_test, y_predict), recall_score(y_test, y_predict)
