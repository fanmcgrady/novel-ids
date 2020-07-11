from rl import classifier as cls
import numpy as np

import random
from utils import load_data,data_preprocessing

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

        self.reset()

    def random_action(self):
        while True:
            action = random.randint(0, self.action_size - 1)
            if action in self.state_index:
                break
        return action

    def step(self, action_index):
        self.state_index.append(action_index)
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
        self.state_index = []
        return self.get_one_hot()

    def get_reward(self):
        temp = [str(x) for x in self.state_index]
        temp = '.'.join(temp)
        reward = self.reward_dict.get(temp, -1)

        if reward == -1:
            reward = self.classifier.classify(self.method, self.state_index)
            # reward是字典
            result = 0
            for element in reward.values():
                result += 0.2*element
            self.add_dict(result)

            reward = result

        return reward

    def add_dict(self, reward):
        temp = [str(x) for x in self.state]
        temp = '.'.join(temp)
        self.reward_dict[temp] = reward

    def get_one_hot(self):
        return np.array([1 if i in self.state_index else 0 for i in range(self.state_size)])