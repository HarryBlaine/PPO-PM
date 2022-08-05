import pickle
from universal.algos import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from env.Portfolio_management import CustomEnv
import warnings
warnings.filterwarnings("ignore")

EPISODES = 1000
lr = 0.001
gamma = 0.99
lmbda = 0.95
epochs = 3
eps_clip = 0.2

class PPO(nn.Module):
    def __init__(self,input_size,action_num):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(input_size, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, action_num)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob,lstm_hidden
    
    def v(self, x, hidden):

        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(epochs):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)

            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage

            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

def main():

    df = pd.read_csv('./Stock Market/FTSE_close.csv')
    df_original_train = df.head(round(0.8*len(df)))
    df_original_test = df.tail(round(0.2*len(df)))

    df_ob = pd.read_csv('./Stock Market/FTSE_final.csv')
    df_ob_train = df_ob.head(round(0.8 * len(df)))
    df_ob_test = df_ob.tail(round(0.2 * len(df)))
    train_set = (df_ob_train - np.mean(df_ob_train.to_numpy())) / np.std(df_ob_train.to_numpy())
    test_set = (df_ob_test - np.mean(df_ob_train.to_numpy())) / np.std(df_ob_train.to_numpy())

    train_set = pd.DataFrame(train_set)
    test_set = pd.DataFrame(test_set)

    train_set = pd.DataFrame(train_set)
    test_set = pd.DataFrame(test_set)

    env = CustomEnv(df_original_train, train_set)
    model = PPO(880*5+1, 15)
    #pkl_file = open('model/third.pkl', 'rb')
    #model = pickle.load(pkl_file)
    n_epi = 0

    total_test_reward = []
    total_train_reward = []

    while(n_epi!= 3000):

        model.train()
        env = CustomEnv(df_original_train,train_set)
        n_epi +=1
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.reset() # observation
        s = s.to_numpy()
        punishment = 0
        s = np.append(s, punishment)
        total_step = len(train_set)//10 - 2
        total_step_test = len(test_set)//10 - 2
        done = False
        t = 0
        train_reward_list = 1
        action_list = []
        while(done == False):

            t = t+1
            h_in = h_out
            prob, h_out = model.pi(torch.from_numpy(s.reshape(-1)).float(), h_in)

            prob = prob.view(-1)

            try:
                m = Categorical(prob)
            except:
                exit()

            action = m.sample().item()
            next_s, r, done, _,result= env.step(action)
            pr = r

            action_list.append(action)
            action_counter = max(np.bincount(action_list))
            most_action = np.argmax(np.bincount(action_list))

            if action == most_action and action_counter > total_step//3:
                r = r - 1
                punishment = 1
                s[len(s) - 1] = punishment
            else:
                punishment = 0
                s[len(s)-1]=punishment

            next_s = next_s.to_numpy()
            next_s = np.append(next_s, punishment)

            model.put_data((s.reshape(-1), action, r , next_s.reshape(-1), prob[action].item(), h_in, h_out ,done))
            s = next_s

            pr = pr / 20
            train_reward_list = train_reward_list * (pr + 1)

            if done:
                break
        print("# of episode :{} steps :{} total_wealth :{}".format(n_epi,t,train_reward_list))
        print("most_action is: " + str(most_action) + "   counter: " + str(action_counter))
        total_train_reward.append(train_reward_list)

        if  n_epi% 100 == 0:
            model_name = './model/'+ str(n_epi) + '.pkl'
            plt.clf() 
            pickle.dump(model, open(model_name, 'wb'))
        model.train_net()
        a = 0
        if n_epi% 10 == 0:
            test_action_list = []
            model.eval()
            env = CustomEnv(df_original_test,test_set)
            s = env.reset()  # observation
            punishment = 0
            s = np.append(s, punishment)
            done = False
            t = 0
            test_reward_list = 1
            while (done == False):
               # s = s.to_numpy()
                t = t + 1
                h_in = h_out

                prob, h_out = model.pi(torch.from_numpy(s.reshape(-1)).float(), h_in)

                prob = prob.view(-1)

                np_array = prob.detach().numpy()

                count = np_array.shape[0]
                for i in range(count):
                    if np_array[i] == np_array.max():
                        action = i
                        break

                next_s, r, done, _, result = env.step(action)

                a = a+1
                r = r / 20
                test_action_list.append(action)
                test_action_counter = max(np.bincount(test_action_list))
                test_most_action = np.argmax(np.bincount(test_action_list))

                if action == test_most_action and test_action_counter > total_step_test // 3:
                    punishment = 1
                    s[len(s) - 1] = punishment
                else:
                    punishment = 0
                    s[len(s) - 1] = punishment

                next_s = np.append(next_s, punishment)
                test_reward_list = test_reward_list * (1+r)
                s = next_s

                if done:
                    break
            total_test_reward.append(test_reward_list)
            print("# Testing!!! total_wealth :{}".format(test_reward_list))
    plt_train_reward = np.array(total_train_reward)

    plt_test_reward = np.array(total_test_reward)

    plt.plot(plt_train_reward)
    p1 = "./image/train_image_3"
    plt.savefig(p1)
    plt.figure()
    plt.plot(plt_test_reward)

    p2 = "./image/test_image_3"
    plt.savefig(p2)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
