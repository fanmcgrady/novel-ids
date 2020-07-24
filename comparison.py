from rl.classifier import Classifier,CLASSIFIER_POOL,CLASSIFIER_POOL_TEST
from utils import load_data,data_preprocessing
import time

data_path = 'dataset/preprocessed/KDDTrain+.csv'
result_path = 'result/'

# 使用全部特征
def full_feature(data_path):
    processed_data, label = load_normalize_data(data_path)
    classifier = Classifier(processed_data, label)
    date = time.strftime('%Y_%m_%d')
    # 统计一下平均数据
    # avg_accuracy = 0
    # avg_precision = 0
    # avg_recall = 0
    # average_f1 = 0
    # avg_false_alarm_rate = 0
    # avg_miss_alarm_rate = 0
    # avg_train_time = 0
    # avg_time_per_sample = 0
    for key in CLASSIFIER_POOL_TEST.keys():
        result = classifier.classify(CLASSIFIER_POOL_TEST.get(key),[22, 28, 15, 18, 38, 34])
        # avg_accuracy += result['Accuracy']
        # avg_precision += result['Precision']
        # avg_recall += result['Recall']
        # average_f1 += result['F1 Score']
        # avg_false_alarm_rate += result['False Alarm Rate']
        # avg_miss_alarm_rate += result['Miss Alarm Rate']
        # avg_train_time += result['Train Time']
        # avg_time_per_sample += result['Test Time For Per Sample']
        result_file = save_result(result, result_path, date, 'NB_Result_', key)

    print('Training complete! The result was saved to '+result_path+result_file)

    # count = len(CLASSIFIER_POOL)
    # with open(result_path+'average_.txt', 'a+') as f:
    #     f.write('----------------------------------------------------------\n')
    #     f.write('Avg Accuracy : {}\n'.format(avg_accuracy/count))
    #     f.write('Avg Precision : {}\n'.format(avg_precision/count))
    #     f.write('Avg Recall : {}\n'.format(avg_recall/count))
    #     f.write('Avg F1 Score : {}\n'.format(average_f1/count))
    #     f.write('Avg False Alarm Rate : {}\n'.format(avg_false_alarm_rate/count))
    #     f.write('Avg Miss Alarm Rate : {}\n'.format(avg_miss_alarm_rate/count))
    #     f.write('Avg Train Time : {}\n'.format(avg_train_time/count))
    #     f.write('Avg Test Time For Per Sample : {}\n'.format(avg_time_per_sample/count))
    #     f.write('-------------Saved time: ' + time.strftime('%Y/%m/%d/%H:%M:%S') + '----------------\n')


'''
2015——Staudemeyer
Feature Set Reduction for Automatic Network Intrusion Detection
with Machine Learning Algorithms
使用了1，2，3，5，6，8，25，33，35，36，40号特征
MLP获得
'''
def Staudemeyer_method(data_path):
    processed_data, label = load_normalize_data(data_path)
    classifier = Classifier(processed_data, label)
    date = time.strftime('%Y_%m_%d')
    feature = [0, 1, 2, 4, 5, 7, 24, 32, 34, 39]
    method = CLASSIFIER_POOL['MLP']
    result = classifier.classify(method, feature)
    result_file = save_result(result, result_path, date, 'Staudemeyer', method)
    print('Training complete! The result was saved to ' + result_path + result_file)



# 读取数据并且范化数据
def load_normalize_data(path):

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
        f.write('False Alarm Rate : {}\n'.format(result['False Alarm Rate']))
        f.write('Miss Alarm Rate : {}\n'.format(result['Miss Alarm Rate']))
        f.write('Train Time : {}\n'.format(result['Train Time']))
        f.write('Test Time For Per Sample : {}\n'.format(result['Test Time For Per Sample']))
        f.write('-------------Saved time: '+time.strftime('%Y/%m/%d/%H:%M:%S')+'----------------\n')

    return result_file


if __name__ == '__main__':
    # 使用全部特征的
    full_feature(data_path=data_path)
    # Staudemeyer_method(data_path=data_path)
