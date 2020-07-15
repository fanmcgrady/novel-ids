import random

import numpy as np


# action space 中的最后一个动作为终止

# 自己构建的环境
class MyEnv:
    def __init__(self, state_size, max, classifier, method):
        self.state_size = state_size
        self.action_size = state_size
        self.max = max  # 最多选取max个特征，超出直接终止
        self.classifier = classifier
        self.method = method
        self.reward_dict = {}
        self.pre_accuracy = 0

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
        reward = self.get_reward()

        # reward = random.random()*100
        return self.get_one_hot(), reward, self.done

    def reset(self):
        self.done = False
        self.state_index = set()
        return self.get_one_hot()

    def get_reward(self):
        temp = [str(x) for x in self.state_index]
        temp = '.'.join(temp)
        if temp in self.reward_dict.keys():
            item = self.reward_dict.get(temp)
            self.pre_accuracy = item[1]
            return item[0]
        else:
            reward = self.classifier.classify(self.method, self.state_index)
            # reward是字典
            # result = 0
            # for element in reward.values():
            #     result += 0.2*element

            accuracy = reward['Accuracy']
            result = -1
            # 越选指标越小的情况
            if self.pre_accuracy > accuracy:
                result = -1
            # 选对的情况
            else:
                result = accuracy

            self.add_dict(result, accuracy)
            self.pre_accuracy = accuracy

            return result

    def add_dict(self, reward, accuracy):
        temp = [str(x) for x in self.state_index]
        temp = '.'.join(temp)
        self.reward_dict[temp] = [reward, accuracy]

    def get_one_hot(self):
        return np.array([1 if i in self.state_index else 0 for i in range(self.state_size)])
