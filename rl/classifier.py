import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import time


CLASSIFIER_POOL = {
    'RandomForest': RandomForestClassifier(random_state=0, n_estimators=50),
    'KNN': KNeighborsClassifier(),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(32,16), solver='adam', alpha=1e-5),
    'Ada': AdaBoostClassifier(n_estimators=100),
    'BAGGING': BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
    # SVM训练太慢，原因是一直没收敛，需要调一下参数
    'SVM': SVC(kernel='rbf', probability=True, gamma='auto', max_iter=3000),
    'GBDT': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
}

CLASSIFIER_POOL_TEST = {'BAGGING':BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)}

class Classifier():
    def __init__(self, data, label):
        self.data = data
        self.label = label


    # def encoding(self):
    #     classifier = ['RandomForest', 'KNN', 'NB', 'DT', 'MLP', 'Ada', 'BAGGING', 'SVM', 'GBDT']
    #     return classifier.index(self.classifier)

    # feature包含选择的特征的index,默认提取全部特征
    def classify(self, classifier, feature=None):

        # 返回的result字典，里面存有分类的各种评价结果
        result = {}
        # 存放提取后的特征
        x = []

        if feature == None:
            x = self.data
        else:
            # 把对应的特征取出来，可能会有重复
            feature = set(feature)
            for data_row in self.data:
                new_data_row = []
                for i in feature:
                    new_data_row.append(data_row[i])
                x.append(new_data_row)

        # 划分训练集、测试集
        x_train, x_test, y_train, y_test = train_test_split(x, self.label, test_size=0.2, random_state=0)

        # 训练
        train_start = time.time()
        classifier.fit(x_train, y_train)
        train_end = time.time()
        train_time = train_end - train_start
        sample_number = len(x_test)


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

        # 测试
        test_start = time.time()
        y_predict = classifier.predict(x_test)
        test_end = time.time()
        test_time = test_end - test_start

        # 获取混淆矩阵，得到各个指标
        cm = metrics.confusion_matrix(y_test, y_predict)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]
        # print('TP : {}, FP : {}, FN : {}, TN :{}'.format(TP, FP, FN, TN))

        accuracy = metrics.accuracy_score(y_test, y_predict)
        precision = metrics.precision_score(y_test, y_predict, pos_label='1', average='binary')
        recall = metrics.recall_score(y_test, y_predict, pos_label='1', average='binary')
        f1_score = metrics.f1_score(y_test, y_predict, pos_label='1', average='binary')
        # 误警率
        false_alarm_rate = FP / (FP + TN)
        # 漏警率
        miss_alarm_rate = FN / (TP + FN)
        # print('FAR: {}'.format(false_Alarm_Rate))
        # print('MAR: {}'.format(miss_ALarm_Rate))

        result['Accuracy'] = accuracy
        result['Precision'] = precision
        result['Recall'] = recall
        result['F1 Score'] = f1_score
        result['False Alarm Rate'] = false_alarm_rate
        result['Miss Alarm Rate'] = miss_alarm_rate
        result['Train Time'] = train_time
        result['Test Time For Per Sample'] = test_time/sample_number


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