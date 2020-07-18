import argparse
import time

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers
from chainerrl import replay_buffer, explorers
from chainerrl.action_value import DiscreteActionValue
import rl.action_value as ActionValue
from rl.agent import MyDoubleDQN
from chainerrl.agents import double_dqn
from chainerrl.replay_buffers import prioritized

import utils
from rl import env as Env
from rl.classifier import CLASSIFIER_POOL

parser = argparse.ArgumentParser()
parser.add_argument('--max-feature', type=int, default=10)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--cls', type=str, default='RandomForest')
args = parser.parse_args()

# 可变参数
feature_number = 41  # 特征总数量
feature_max_count = args.max_feature  # 选取的特征数目大于该值时，reward为0，用于当特征数目在该范围内时，成功率最多可以到达多少
MAX_EPISODE = 1000
net_layers = [32, 16]

result_file = 'result/result-{}-{}.txt'.format(args.cls, time.strftime('%Y%m%d%H%M'))

# 每一轮逻辑如下
# 1. 初始化环境，定义S和A两个list，用来保存过程中的state和action。进入循环，直到当前这一轮完成（done == True）
# 2. 在每一步里，首先选择一个action，此处先用简单的act()代替
# 3. 接着env接收这个action，返回新的state，done和reward，当done==False时，reward=0，当done==True时，reward为模型的准确率
# 4. 如果done==True，那么应该把当前的S、A和reward送到replay buffer里（replay也应该在此时进行），往replay buffer里添加时，
#    每一对state和action都有一个reward，这个reward应该和env返回的reward（也就是该模型的acc）和count有关。

# 用这个逻辑替代原来的my_train的逻辑，只需要把agent加入即可，agent应该是不需要修改的

def main():
    episode_reward = []

    class QFunction(chainer.Chain):
        def __init__(self, obs_size, n_actions, n_hidden_channels=None):
            super(QFunction, self).__init__()
            if n_hidden_channels is None:
                n_hidden_channels = net_layers
            net = []
            inpdim = obs_size
            for i, n_hid in enumerate(n_hidden_channels):
                net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
                # net += [('norm{}'.format(i), L.BatchNormalization(n_hid))]
                net += [('_act{}'.format(i), F.relu)]
                net += [('_dropout{}'.format(i), F.dropout)]
                inpdim = n_hid

            net += [('output', L.Linear(inpdim, n_actions))]

            with self.init_scope():
                for n in net:
                    if not n[0].startswith('_'):
                        setattr(self, n[0], n[1])

            self.forward = net

        def __call__(self, x, test=False):
            """
            Args:
                x (ndarray or chainer.Variable): An observation
                test (bool): a flag indicating whether it is in test mode
            """
            for n, f in self.forward:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                elif n.startswith('_dropout'):
                    x = f(x, 0.1)
                else:
                    x = f(x)

            return ActionValue.DiscreteActionValue(x)

    def evaluate(env, agent, current):
        for i in range(1):
            print("evaluate episode: {}".format(current))
            state = env.reset()
            terminal = False
            action_list = []

            while not terminal:
                action = agent.act(state)
                action_list.append(action)
                state, reward, terminal = env.step(action)

                if terminal or len(action_list) > 20:
                    if len(action_list) > 20:
                        terminal = True

                    with open(result_file, 'a+') as f:
                        f.write(
                            "--------------------------------------------------------------------------------------------------\n"
                            "evaluate episode:{}, reward = {}, action = {}\n"
                            "-------------------------------------------------------------------------------------------------\n"
                                .format(current, reward, action_list)
                        )
                        print(
                            "--------------------------------------------------------------------------------------------------\n"
                            "evaluate episode:{}, reward = {}, action = {}\n"
                            "-------------------------------------------------------------------------------------------------\n"
                                .format(current, reward, action_list)
                        )

    # 开始训练
    # def train_agent(env, agent):
    #     train_agent_with_evaluation(
    #         agent, env,
    #         steps=10000,  # Train the graduation_agent for this many rounds steps
    #         max_episode_len=MAX_EPISODE,  # Maximum length of each episodes
    #         eval_interval=args.eval_interval,  # Evaluate the graduation_agent after every 1000 steps
    #         eval_n_runs=args.eval_n_runs,  # 100 episodes are sampled for each evaluation
    #         outdir='result',  # Save everything to 'result' directory
    #     )
    #
    #     return env, agent

    def train_agent(env, agent):
        for episode in range(MAX_EPISODE):
            state = env.reset()
            terminal = False
            reward = 0
            t = 0
            action_list = []
            while not terminal:
                t += 1
                action = agent.act_and_train(
                    state, reward)  # 此处action是否合法（即不能重复选取同一个指标）由agent判断。env默认得到的action合法。
                action_list.append(action)
                state, reward, terminal = env.step(action)
                print("episode:{}, t:{}, action:{}, reward = {}".format(episode, t, action_list, reward))

                if terminal:
                    with open(result_file, 'a+') as f:
                        f.write("train episode:{}, reward = {}, action = {}\n"
                                .format(episode, reward, action_list))
                        print("train episode:{}, reward = {}, action = {}\n"
                                .format(episode, reward, action_list))

                        agent.stop_episode()
                        episode_reward.append(reward)
                        if (episode + 1) % 2 == 0 and episode != 0:
                            evaluate(env, agent, (episode + 1) / 2)

    def create_agent(env):
        state_size = env.state_size
        action_size = env.action_size
        q_func = QFunction(state_size, action_size)

        start_epsilon = 1.
        end_epsilon = 0.3
        decay_steps = 20
        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon, end_epsilon, decay_steps,
            env.random_action)

        opt = optimizers.Adam()
        opt.setup(q_func)

        # rbuf_capacity = 5 * 10 ** 3
        minibatch_size = 16

        # steps = 1000
        replay_start_size = 20
        update_interval = 5
        # betasteps = (steps - replay_start_size) // update_interval
        # rbuf = replay_buffer.PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)
        rbuf = prioritized.PrioritizedReplayBuffer()

        phi = lambda x: x.astype(np.float32, copy=False)

        # agent = double_dqn.DoubleDQN(q_func,
        #                              opt,
        #                              rbuf,
        #                              gamma=0.99,
        #                              explorer=explorer,
        #                              replay_start_size=replay_start_size,
        #                              target_update_interval=10,  # target q网络多久和q网络同步
        #                              update_interval=update_interval,
        #                              phi=phi,
        #                              minibatch_size=minibatch_size,
        #                              gpu=args.gpu,  # 设置是否使用gpu
        #                              episodic_update_len=16)


        # 自己的DDQN
        agent = MyDoubleDQN(q_func,
                            opt,
                            rbuf,
                            gamma=0.99,
                            explorer=explorer,
                            replay_start_size=replay_start_size,
                            target_update_interval=10,  # target q网络多久和q网络同步
                            update_interval=update_interval,
                            phi=phi,
                            minibatch_size=minibatch_size,
                            gpu=args.gpu,  # 设置是否使用gpu
                            episodic_update_len=16)

        return agent

    def train():
        env = Env.MyEnv(feature_number,
                        feature_max_count,
                        utils.get_classifier(),
                        CLASSIFIER_POOL[args.cls])
        agent = create_agent(env)
        train_agent(env, agent)

        # evaluate(env, agent)

        return env, agent

    train()

    # 用于计算本次训练中最大的准确率以及平均准确率
    max_reward = max(episode_reward)
    average_reward = 0
    for i in range(len(episode_reward) - 1):
        average_reward = average_reward + episode_reward[i]
    average_reward = average_reward / len(episode_reward)

    # 写入文件的最后一行
    with open(result_file, 'a+') as f:
        f.write(
            "The max reward of this train:{}, the average reward of this train:{}"
                .format(max_reward, average_reward))


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print("elapsed: {}".format(elapsed))
    # 训练时间
    with open(result_file, 'a+') as f:
        f.write("Training elapsed:{} seconds".format(elapsed))
