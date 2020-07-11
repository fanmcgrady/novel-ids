from rl.classifier import Classifier,CLASSIFIER_POOL
from utils import load_data,data_preprocessing
import time

data_path = 'dataset/preprocessed/KDDTrain+.csv'
result_path = 'result/'

# 使用全部特征
def full_feature(data_path):
    data, label = load_data(data_path)
    processed_data = data_preprocessing(data)
    print(type(data))
    print(type(label))
    print(label)
    classifier = Classifier(processed_data, label)
    date = time.strftime('%Y_%m_%d')
    result_file = 'full_feature_result_'+date+'.txt'
    with open(result_path + result_file, 'w+') as f:
        f.write('Feature used: All feature\n')
        for key in CLASSIFIER_POOL.keys():
            result = classifier.classify(CLASSIFIER_POOL.get(key))
            f.write('----------------------------------------------------------\n')
            f.write('Classifier : {}\n'.format(key))
            f.write('Accuracy : {}\n'.format(result['Accuracy']))
            f.write('Precision : {}\n'.format(result['Precision']))
            f.write('Recall : {}\n'.format(result['Recall']))
            f.write('F1 Score : {}\n'.format(result['F1 Score']))
            f.write('Train Time : {}\n'.format(result['Train Time']))
            f.write('Test Time For Per Sample : {}\n'.format(result['Test Time For Per Sample']))
            f.write('----------------------------------------------------------\n')
        f.write('Saved time: '+time.strftime('%Y/%m/%d/%H:%M:%S'))
    print('Training complete! The result was saved to '+result_path+result_file)

# xx的方法,使用了
def method():
    pass


if __name__ == '__main__':
    #full_feature(data_path=data_path)
    pass