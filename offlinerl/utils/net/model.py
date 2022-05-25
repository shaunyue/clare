import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
# LOG_SIG_MIN = -5
epsilon = 1e-6
MEAN_MIN = -9.0
MEAN_MAX = 9.0
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        # self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, 1)
        # self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, 1)
        self.identify=nn.Identity()
        self.relu = Swish()
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(weights_init_)

    '''
    改动了激活函数
    '''
    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = self.relu(self.linear1(xu))
        x1 = self.relu(self.linear2(x1))
        # x1 = F.relu(self.linear3(x1))
        # x1 = self.LeakyRelu(self.linear1(xu))
        # x1 = self.LeakyRelu(self.linear2(x1))
        
        x1 = self.linear4(x1)
        # x1= self.identify(x1)
        # x2 = F.relu(self.linear4(xu))
        # x2 = F.relu(self.linear5(x2))
        # x2 = self.LeakyRelu(self.linear4(xu))
        # x2 = self.LeakyRelu(self.linear5(x2))
        
        # x2 = self.linear6(x2)
        x2 = self.relu(self.linear5(xu))
        x2 = self.relu(self.linear6(x2))
        # x2 = self.relu(self.linear3(x2))
        # x1 = self.LeakyRelu(self.linear1(xu))
        # x1 = self.LeakyRelu(self.linear2(x1))
        
        x2 = self.linear7(x2)
        # x2 = self.identify(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = Swish()
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        self.flag=action_space
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            # print("space")
            # # print(action_space.shape)
            # # print(num_actions)
            # self.action_scale = torch.FloatTensor(
            #     (action_space.high - action_space.low) / 2.)
            # self.action_bias = torch.FloatTensor(
            #     (action_space.high + action_space.low) / 2.)
            # print("space")
            # print(self.action_scale)
            # print(num_actions)
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # 对log_st设置了上下限
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def atanh(self,x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # mean std 分别为方差和均值
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # x_t = normal.sample()
        y_t = torch.tanh(x_t)
        # print("self.action_scale")
        tempx=torch.ones(self.action_scale.shape,device="cuda:0")
        if(torch.norm(tempx-self.action_scale)!=0):
            print(self.action_scale)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

        # mean, log_std = self.forward(state)
        # std = log_std.exp()
        # # mean std 分别为方差和均值
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # # x_t = normal.sample()
        # x_t.requires_grad_()
        # y_t = torch.tanh(x_t)
        # action = y_t
        # # mean, log_std = self.forward(y_t)
        # # mean = torch.clamp(mean, self.MEAN_MIN, self.MEAN_MAX)
        # # std = log_std.exp()
        # # mean std 分别为方差和均值
        # # normal = Normal(mean, std)
        # log_prob = normal.log_prob(x_t) - torch.log(
        #     1 - y_t * y_t + epsilon
        # )
        # log_prob = log_prob.sum(1, keepdim=True)
        # # print("self.action_scale",self.action_scale)
        # action1 = y_t
        # # if(torch.norm(action1-action)!=0):
        #     # print(torch.norm(action1-action))
        # # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        
        # mean = torch.tanh(mean)
        # return action, log_prob, mean

    def sample_com(self):
        # mean std 分别为方差和均值
        normal = Normal(self.mean, self.std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # x_t = normal.sample()
        y_t = torch.tanh(x_t)
        # print("self.action_scale")
        # print(self.action_scale)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
