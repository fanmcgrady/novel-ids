from rl.classifier import Classifier,CLASSIFIER_POOL
from utils import load_data,data_preprocessing
import time

data_path = 'dataset/preprocessed/KDDTrain+.csv'
result_path = 'result/'

# 使用全部特征
def full_feature(data_path):
    processed_data, label = load_minmax(data_path)
    classifier = Classifier(processed_data, label)
    date = time.strftime('%Y_%m_%d')
    for key in CLASSIFIER_POOL.keys():
        result = classifier.classify(CLASSIFIER_POOL.get(key))
        result_file = save_result(result, result_path, date, 'Full_feature', key)
    print('Training complete! The result was saved to '+result_path+result_file)

'''
2015——Staudemeyer
Feature Set Reduction for Automatic Network Intrusion Detection
with Machine Learning Algorithms
使用了1，2，3，5，6，8，25，33，35，36，40号特征
MLP获得
'''
def Staudemeyer_method(data_path):
    processed_data, label = load_minmax(data_path)
    classifier = Classifier(processed_data, label)
    date = time.strftime('%Y_%m_%d')
    feature = [0, 1, 2, 4, 5, 7, 24, 32, 34, 39]
    method = CLASSIFIER_POOL['MLP']
    result = classifier.classify(method, feature)
    result_file = save_result(result, result_path, date, 'Staudemeyer', method)
    print('Training complete! The result was saved to ' + result_path + result_file)

# 读取数据并且范化数据
def load_minmax(path):

    data, label = load_data(path)
    processed_data = data_preprocessing(data)

    return processed_data, label

# 保存结果
def save_result(result, result_path, date, compare_object, method):
    result_file = compare_object + date + '.txt'
    with open(result_path + result_file, 'a+') as f:
        f.write('----------------------------------------------------------\n')
        f.write('Classifier : {}\n'.format(method))
        f.write('Accuracy : {}\n'.format(result['Accuracy']))
        f.write('Precision : {}\n'.format(result['Precision']))
        f.write('Recall : {}\n'.format(result['Recall']))
        f.write('F1 Score : {}\n'.format(result['F1 Score']))
        f.write('Train Time : {}\n'.format(result['Train Time']))
        f.write('Test Time For Per Sample : {}\n'.format(result['Test Time For Per Sample']))
        f.write('-------------Saved time: '+time.strftime('%Y/%m/%d/%H:%M:%S')+'----------------\n')

    return result_file




if __name__ == '__main__':
    # 使用全部特征的
    full_feature(data_path=data_path)
    # Staudemeyer_method(data_path=data_path)
    # pass