
import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt
import time

import multiprocessing

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('Pendulum-v1').unwrapped
num_state = env.observation_space.shape[0]#3
num_action = env.action_space.shape[0]#1
torch.manual_seed(args.seed)
env.seed(args.seed)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord',['episode', 'reward'])

class Actor(nn.Module):     # 可替换部分
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64,8)
        self.mu_head = nn.Linear(8, 1)
        self.sigma_head = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        mu = self.mu_head(x)
        sigma = self.sigma_head(x)

        return mu, sigma

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value= nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

class PPO():     # 需要修改升级部分
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1600#1000
    batch_size = 128#8

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().float()
        self.critic_net = Critic().float()
        #self.actor_net = torch.nn.DataParallel(self.actor_net)
        #self.critic_net = torch.nn.DataParallel(self.critic_net)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 4e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor_net(state)
        dist = Normal(mu, torch.abs(sigma))# sigma must >0
        action = dist.sample()# scalar
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net'+str(time.time())[:10],+'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net'+str(time.time())[:10],+'.pkl')

    def store_transition(self, transition):
        if type(transition) != list:
            self.buffer.append(transition)
        else:
            self.buffer.extend(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step +=1
        #print(self.buffer)
        state = torch.tensor(np.array([t.state for t in self.buffer ]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1)
        next_state = torch.tensor(np.array([t.next_state for t in self.buffer]), dtype=torch.float)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1)

        reward = (reward - reward.mean())/(reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net(next_state)

        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                # epoch iteration, PPO core!!!
                mu, sigma = self.actor_net(state[index])
                n = Normal(mu, sigma.abs())# sigma must >0
                action_log_prob = n.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                
                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]
        
class AgentPPO(PPO):   # 保持
    def __init__(self,net):
        super(AgentPPO,self).__init__()
        self.buffer_capacity = 200
        self.actor_net = net.actor_net
        self.critic_net = net.critic_net

    def update_agent(self):
        score = 0
        state = env.reset()
        for t in range(200):   # 每局游戏执行多少次操作
            action, action_log_prob = self.select_action(state) # float,float
            next_state, reward, done, info = env.step(action)
            trans = Transition(state, action, reward, action_log_prob, next_state)
            #if args.render: env.render()
            score += reward
            state = next_state
            if self.store_transition(trans):
                #print(True)
                #agent.update()
                return self.buffer,score
        
            
    

def child_process2(pipe): # 保持
    
    # env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Pendulum-v1').unwrapped    # 创建游戏

    env.reset()
    while True:
        net = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
        ppo = AgentPPO(net)
        transitions = ppo.update_agent()
        
        """pipe不能直接传输buffer回主进程，可能是buffer内有transition，因此将数据取出来打包回传"""
        pipe.send(transitions)

def main():

    root_agent = PPO()
    child_agent = AgentPPO(root_agent)
    process_num = 8        # 需要测试最优配比是多少
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(target=child_process2, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    [pipe_dict[i][0].send(child_agent) for i in range(process_num)]
    [p.start() for p in child_process_list]
    
    training_records = []
    running_reward = -1000

    for i_epoch in range(1000):
        print(i_epoch)
        score = 0
        state = env.reset()
        if args.render: env.render()
        buffer_list = []
        for i in range(process_num):
            receive = pipe_dict[i][0].recv()        # 这句带同步子进程的功能，收不到子进程的数据就都不会走到for之后的语句
            trans = receive[0]
            buffer_list.extend(trans)
            #reward += receive[1]
            score +=receive[1]
        root_agent.store_transition(buffer_list)
        root_agent.update()
        child_agent.actor_net.load_state_dict(root_agent.actor_net.state_dict())
        child_agent.critic_net.load_state_dict(root_agent.critic_net.state_dict())
        [pipe_dict[i][0].send(child_agent) for i in range(process_num)]
        
        # for t in range(200):
        #     action, action_log_prob = agent.select_action(state)# float,float
        #     next_state, reward, done, info = env.step(action)
        #     trans = Transition(state, action, reward, action_log_prob, next_state)
        #     if args.render: env.render()
        #     if agent.store_transition(trans):
        #         print(True)
        #         agent.update()
        #     score += reward
        #     state = next_state
        #print(t)
        
        score = score/process_num
        running_reward = running_reward * 0.9 + score * 0.1
        #training_records.append(TrainingRecord(i_epoch, running_reward))
        if i_epoch % 10 ==0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
        if running_reward > -200:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            root_agent.save_param()
            break

if __name__ == '__main__':
    main()
