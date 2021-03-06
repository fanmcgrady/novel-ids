import csv
import os
import numpy as np
from sklearn import preprocessing
from rl.classifier import Classifier, CLASSIFIER_POOL

protocol_type = ['tcp', 'udp', 'icmp']
service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo',
           'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
           'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
           'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
           'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
           'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
           'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']

data_path = 'dataset/preprocessed/KDDTrain+.csv'
result_path = 'result/'

def get_classifier():
    data, label = load_data(data_path)
    processed_data = data_preprocessing(data)
    return Classifier(processed_data, label)

# 读取csv,针对两种不同的csv文件
def load_csv(path, original=False):
    data = []
    label = []
    if original == True:
        index = 2
    else:
        index = 1
    with open(path) as csvfile:
        for row in csvfile:
            row = row.split(',')
            # 处理protocol_type
            row[1] = str(protocol_type.index(row[1]))
            # 处理service
            row[2] = str(service.index(row[2]))
            # 处理flag
            row[3] = str(flag.index(row[3]))
            # 将str类型转化为float类型
            for i in range(len(row)-index):
                row[i] = float(row[i])
            # 处理label
            row[-index] = row[-index].replace('\n', '')
            if row[-index] == 'normal':
                label.append(0)
            else:
                label.append(1)
            data.append(row[:-index])

    return data, label

# 保存csv
def save_csv(path, data, label):

    if os.path.exists(path):
        os.remove(path)

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            data[i].append(label[i])
            writer.writerow(data[i])

    print("Save data successfully to: {}".format(path))

# 数据预处理
def data_preprocessing(data):

    # Min-Max数据
    min_max_scaler = preprocessing.MinMaxScaler()
    # temp_data = min_max_scaler.fit_transform(data)
    # 正则化数据,这个效果回比上面Min-Max好一些
    preprocessed_data = preprocessing.normalize(data, norm='l2')

    return preprocessed_data
    # return preprocessed_data

def load_data(path):

    data = []
    label = []

    with open(path) as f:
        for row in f:
            row = row.split(',')
            data.append(row[:-1])
            # 去掉换行
            temp = row[-1].replace('\n', '')
            label.append(temp)

    return data, label


if __name__ == '__main__':
    # path = "dataset/KDDTrain+.csv"
    # data, label = load_csv(path)
    # classifier = Classifier(data, label)
    # for key in CLASSIFIER_POOL.keys():
    #     result = classifier.classify(CLASSIFIER_POOL.get(key))
    #     print('----------------------------------------------------------')
    #     print('Classifier : {}'.format(key))
    #     print('Accuracy : {}'.format(result['Accuracy']))
    #     print('Precision : {}'.format(result['Precision']))
    #     print('Recall : {}'.format(result['Recall']))
    #     print('F1 Score : {}'.format(result['F1 Score']))
    #     print('Train Time : {}'.format(result['Train Time']))
    #     print('Test Time For Per Sample : {}'.format(result['Test Time For Per Sample']))
    #     print('----------------------------------------------------------')
    # data, label = load_csv(data_path)
    # save_csv(save_path, data, label)
    path = data_path = 'dataset/preprocessed/KDDTrain+.csv'
    data, label = load_data(path)
