import csv
import os
import numpy as np
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

def data_process():
    pass


# 读取csv
def load_csv(data_path):
    data = []
    label = []

    with open(data_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # 处理protocol_type
            row[1] = str(protocol_type.index(row[1]))
            # 处理service
            row[2] = str(service.index(row[2]))
            # 处理flag
            row[3] = str(flag.index(row[3]))
            # 处理label

            if row[-2] == 'normal':
                row[-2] = '0'
            else:
                row[-2] = '1'

            data.append(row[:-2])
            label.append(row[-2])
    data = np.array(data, dtype='float')
    return data, label


# 归一化
def normalization(data):
    for i, d in enumerate(data):
        pass
    return data


# 保存csv
def save_csv(data_path, data):
    if os.path.exists(data_path):
        os.remove(data_path)

    with open(data_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            if data[i]:
                writer.writerow(data[i])
    print("save data successfully to: {}".format(data_path))


if __name__ == '__main__':
    path = "dataset/KDDTrain+.csv"
    data, label = load_csv(path)
    classifier = Classifier(data, label)
    for key in CLASSIFIER_POOL.keys():
        print("{} = {}".format(key, classifier.classify(CLASSIFIER_POOL.get(key))))
