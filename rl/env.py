import random

import numpy as np


# action space 中的最后一个动作为终止

# 自己构建的环境
class MyEnv:
    def __init__(self, state_size, action_size, max, classifier, method):
        self.state_size = state_size
        self.action_size = action_size
        self.max = max  # 最多选取max个特征，超出直接终止
        self.classifier = classifier
        self.method = method
        self.reward_dict = {}
        self.average = self._average_training_data()

        self.reset()

    def random_action(self):
        while True:
            action = random.randint(0, self.action_size - 1)
            if action in self.state_index:
                continue
            else:
                break
        return action

    def step(self, action_index):
        self.state_index.add(action_index)
        if len(self.state_index) == self.max:  # 已经到达选择数量上线
            self.done = True

            # reward 默认为0
            # if current_count>self.max:
            #     reward = self.max - current_count
            # else:
        reward, classify_result = self.get_reward()

        # reward = random.random()*100
        return self.get_one_hot(classify_result), reward, classify_result, self.done

    def reset(self):
        self.done = False
        self.state_index = set()
        # 记录前一轮的准确率、检测率、以及召回率
        self.pre_accuracy = 0
        self.pre_precision = 0
        self.pre_recall = 0
        self.current_result = {}

        return self.get_one_hot(self.current_result)

    def get_reward(self):
        temp = [str(x) for x in self.state_index]
        temp = '.'.join(temp)
        if temp in self.reward_dict.keys():
            item = self.reward_dict.get(temp)
            self.pre_accuracy = item[1]['Accuracy']
            self.pre_precision = item[1]['Precision']
            self.pre_recall = item[1]['Recall']
            return item[0], item[1]
        else:
            # 获得分类结果的字典
            classify_result = self.classifier.classify(self.method, self.state_index)
            # for element in reward.values():
            #     result += 0.2*element

            accuracy = classify_result['Accuracy']
            precision = classify_result['Precision']
            recall = classify_result['Recall']
            time = classify_result['Test Time For Per Sample']

            # 公式:reward = r_a * 0.4 + r_p * 0.2 + r_r * 0.2 + r_t * 0.2

            # 看看各个指标的情况

            # 准确率
            # 增加了一个feature反而减小了
            if self.pre_accuracy > accuracy:
                r_a = -2
            # 准确率增大
            else:
                if accuracy < 0.80:
                    r_a = 0
                elif accuracy < 0.95:
                    r_a = 0.5
                else:
                    r_a = 1

            # 检测率
            # 增加了一个feature反而减小了
            if self.pre_precision > precision:
                r_p = -2
            # 检测率增大
            else:
                if precision < 0.80:
                    r_p = 0
                elif precision < 0.95:
                    r_p = 0.5
                else:
                    r_p = 1

            # 召回率
            # 增加了一个feature反而减小了
            if self.pre_recall > recall:
                r_r = -2
            # 召回率增大
            else:
                if recall < 0.80:
                    r_r = 0
                elif recall < 0.95:
                    r_r = 0.5
                else:
                    r_r = 1

            # 训练时间,如果比平均时间还短,那么奖励值为0
            if time > 5.43e-5:
                r_t = 0
            elif time > 1.00e-5:
                r_t = 0.5
            else:
                r_t = 1

            # 奖励值权重公式
            reward = r_a * 0.5 + r_p * 0.2 + r_r * 0.2 + r_t * 0.1


            self.add_dict(reward, classify_result)
            self.pre_accuracy = accuracy
            self.pre_precision = precision
            self.pre_recall = recall

            return reward, classify_result

    # key:选取的哪些特征, 形如[1,3,5..]   value:(回报，分类结果)
    def add_dict(self, reward, classify_result):
        temp = [str(x) for x in self.state_index]
        temp = '.'.join(temp)
        self.reward_dict[temp] = [reward, classify_result]

    def get_one_hot(self, current_result):
        # 1、使用0或1
        # state = [1 if i in self.state_index else 0 for i in range(41)]
        # 2、使用平均值
        one_hot_state = [1 if i in self.state_index else 0 for i in range(self.state_size)]
        state = [self.average[i] if one_hot_state[i] > 0 else 0 for i in range(len(one_hot_state))]

        # 3、使用选定特征的平均值+补0
        state = []
        for i in self.state_index:
            state.append(self.average[i])

        for i in range(10 - len(self.state_index)):
            state.append(0)

        count = len(self.state_index)
        accuracy = current_result.get('Accuracy', 0)
        precision = current_result.get('Precision', 0)
        recall = current_result.get('Recall', 0)
        f1_score = current_result.get('F1 Score', 0)
        false_alarm_rate = current_result.get('False Alarm Rate', 0)
        miss_alarm_rate = current_result.get('Miss Alarm Rate', 0)
        time_per_sample = current_result.get('Test Time For Per Sample',0)

        # state.append(count)
        # state.append(accuracy/100)
        # state.append(precision/100)
        # state.append(recall/100)
        # state.append(f1_score/100)
        # state.append(false_alarm_rate)
        # state.append(miss_alarm_rate)
        # state.append(time_per_sample)

        return np.array(state)
        # return np.array(one_hot_state)

    def _average_training_data(self):
        data = self.classifier.data
        average = [0 for _ in range(41)]
        for line in data:
            for i in range(len(line)):
                average[i] += line[i]

        return [item / len(data) for item in average]