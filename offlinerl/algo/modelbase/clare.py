import numpy as np
import torch
import gym
import math
import torch.nn as nn
from torch.functional import F
from torch import distributions as pyd
from torch.utils import data
from torch.optim import Adam
from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import Swish
from offlinerl.utils.exp import setup_seed
from offlinerl.utils.data import SampleBatch
from loguru import logger
from tianshou.data import Batch
from collections import OrderedDict
from tqdm import tqdm


is_with_reward = False


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def orthogonal_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def comp_uncertainty(args, buffer, transition):
    obs = buffer['obs']
    obs = torch.tensor(obs, device=args['device'])
    act = torch.tensor(buffer['act'], device=args['device'])
    obs_act = torch.cat([obs, act], dim=-1)
    next_obs_dists = transition(obs_act)
    if args['uncertainty_mode'] == 'disagreement':
        next_obses_mode = next_obs_dists.mean[:, :, :-1]
        next_obs_mean = torch.mean(next_obses_mode, dim=0)
        diff = next_obses_mode - next_obs_mean
        disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
        uncertainty = disagreement_uncertainty
    else:
        aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
        uncertainty = aleatoric_uncertainty
    return uncertainty


def comp_uncertainty_batch(args, buffer, transition):
    uncertainty = torch.zeros(buffer.shape)
    batch_size = int(args['data_collection_per_epoch'])
    for i in range(0, buffer.shape[0], batch_size):
        if i + batch_size < buffer.shape[0]:
            uncertainty[i: i + batch_size] = comp_uncertainty(args, buffer[i: i + batch_size], transition)
        else:
            uncertainty[i: buffer.shape[0]] = comp_uncertainty(args, buffer[i: buffer.shape[0]], transition)
    return uncertainty, torch.max(uncertainty)


def comp_weight(args, train_buffer, expert_buffer, transition, avg_uncertainty, max_uncertainty):
    whole_buffer = SampleBatch(Batch.cat([train_buffer, expert_buffer]))
    whole_uncertainty, _ = comp_uncertainty_batch(args, whole_buffer, transition)
    whole_uncertainty = whole_uncertainty / max_uncertainty
    train_uncertainty, _ = comp_uncertainty_batch(args, train_buffer, transition)
    train_uncertainty = train_uncertainty / max_uncertainty
    expert_uncertainty, _ = comp_uncertainty_batch(args, expert_buffer, transition)
    expert_uncertainty = expert_uncertainty / max_uncertainty
    avg_uncertainty = avg_uncertainty / max_uncertainty
    num_whole_buffer = whole_buffer.shape[0]
    num_expert_buffer = expert_buffer.shape[0]
    train_weight = torch.zeros(train_uncertainty.shape)
    expert_weight = torch.zeros(expert_uncertainty.shape)
    num_data_large_uncertainty = 1
    num_data_small_uncertainty = 1
    if not args['use_clare_regularization']:
        if args['is_avg_based']:
            train_weight[train_uncertainty < avg_uncertainty - args['u']] = args['beta']
            expert_weight[expert_uncertainty < avg_uncertainty - args['u']] = args['beta']
            expert_weight[expert_uncertainty > avg_uncertainty + args['u']] = - args['beta']
        else:
            train_weight[train_uncertainty < args['u']] = args['beta']
            expert_weight[expert_uncertainty < args['u']] = args['beta']
            expert_weight[expert_uncertainty > args['u']] = - args['beta']
    else:
        if args['is_avg_based']:
            num_data_large_uncertainty = torch.sum(expert_uncertainty > avg_uncertainty + args['u'])
            num_data_small_uncertainty = torch.sum(whole_uncertainty < avg_uncertainty - args['u'])
            train_weight[train_uncertainty < avg_uncertainty - args['u']] = \
                (num_whole_buffer / num_expert_buffer) / num_data_small_uncertainty * num_data_large_uncertainty
            expert_weight[expert_uncertainty < avg_uncertainty - args['u']] = \
                (num_whole_buffer / num_expert_buffer) / num_data_small_uncertainty * num_data_large_uncertainty
            expert_weight[
                expert_uncertainty > avg_uncertainty + args['u']] = - num_whole_buffer / num_expert_buffer
        else:
            num_data_large_uncertainty = torch.sum(expert_uncertainty > args['u'])
            num_data_small_uncertainty = torch.sum(whole_uncertainty < args['u'])
            train_weight[train_uncertainty < args['u']] = \
                (num_whole_buffer / num_expert_buffer) / num_data_small_uncertainty * num_data_large_uncertainty
            expert_weight[expert_uncertainty < args['u']] = \
                (num_whole_buffer / num_expert_buffer) / num_data_small_uncertainty * num_data_large_uncertainty
            expert_weight[expert_uncertainty > args['u']] = - num_whole_buffer / num_expert_buffer
    return train_weight, expert_weight, num_data_large_uncertainty, num_data_small_uncertainty, \
           (num_whole_buffer / num_expert_buffer) / num_data_small_uncertainty * num_data_large_uncertainty, - num_whole_buffer / num_expert_buffer


def merge_data_weight(batch, weight):
    b = {}
    weight_arr = weight.numpy()
    b['obs'] = batch['obs']
    b['act'] = batch['act']
    b['obs_next'] = batch['obs_next']
    b['done'] = batch['done']
    b['rew'] = batch['rew']
    b['weight'] = weight_arr
    merge_batch = Batch(b)
    return merge_batch


def algo_init(args):
    logger.info('Run algo_init function')
    setup_seed(args['seed'])
    if "diverse_data" in args.keys():
        from offlinerl.utils.env import get_env_shape
        obs_shape, action_shape, ac_shape = get_env_shape(args['diverse_data'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'],
                                    args['transition_init_num']).to(args['device'])
    transition_optim = torch.optim.Adam(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)
    reward_function = EnsembleReward(obs_shape, action_shape, args['hidden_layer_size_reward'],
                                     args['reward_layers'], args['reward_init_num'],
                                     use_restr_for_output=args['use_restr_for_output'],
                                     use_dropout=args['use_dropout']).to(args['device'])
    reward_optim = torch.optim.Adam(reward_function.parameters(), lr=args['reward_lr'], weight_decay=1e-5)
    args_sac = {
        "target_update_interval" : args['sac_target_update_interval'],
        "updates_per_step" : args['sac_updates_per_step'],
        "hidden_size" : args['hidden_size'],
        "batch_size" : args['sac_batch_size'],
        "seed" : args['seed'],
        "automatic_entropy_tuning" : args['sac_automatic_entropy_tuning'],
        "alpha" : args['sac_alpha'],
        "eval" : args['sac_eval'],
        "gamma" : args['sac_gamma'],
        "tau" : args['sac_tau'],
        "lr" : args['sac_lr']
    }
    agent_sac = SAC(obs_shape, ac_shape, args_sac)
    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["sac_lr"])

    return {
        "transition": {"net": transition, "opt": transition_optim},
        "reward_function": {"net": reward_function, "opt": reward_optim},
        "log_alpha": {"net": log_alpha, "opt": alpha_optimizer},
        "sac": {"sac": agent_sac},
        "args_sac":{"args_sac":args_sac}
    }


def soft_clamp(x: torch.Tensor, _min=None, _max=None):
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.relu = Swish()
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.relu(self.linear1(xu))
        x1 = self.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = self.relu(self.linear4(xu))
        x2 = self.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):


        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):


        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, _,
                 log_std_bounds, action_space):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.relu = Swish()
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self.outputs = dict()
        self.apply(orthogonal_init_)

    def forward(self, obs):
        x = self.relu(self.linear1(obs))
        x = self.relu(self.linear2(x))
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist

    @staticmethod
    def atanh(x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def compute_log_prob(self, state, action):
        dist = self.forward(state)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DiagGaussianActor, self).to(device)


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.action_range = [action_space.low.min(), action_space.high.max()]
        self.target_update_interval = args["target_update_interval"]
        self.automatic_entropy_tuning = args["automatic_entropy_tuning"]
        self.device = torch.device("cuda")
        self.critic = QNetwork(num_inputs, action_space.shape[0], args['hidden_size']).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args['lr'])
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args['hidden_size']).to(self.device)
        hard_update(self.critic_target, self.critic)
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args['lr'])
        self.policy = DiagGaussianActor(num_inputs, action_space.shape[0], args['hidden_size'], 2, [-5, 2],
                                        action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args['lr'])

    def get_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.clamp(*self.action_range)
        return action.detach().cpu().numpy()[0]

    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.clamp(*self.action_range)
        return action.to(self.device)[0]

    @staticmethod
    def _sync_weight(net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    def update_parameters(self, memory, batch_expert=None, regularization_weight=None):

        state_batch = memory['obs']
        action_batch = memory['act']
        next_state_batch = memory['obs_next']
        reward_batch = memory['rew']
        mask_batch = memory['done']
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_batch,
                               action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if batch_expert is None:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        else:
            obs_expert = batch_expert['obs']
            act_expert = batch_expert['act']
            obs_expert = obs_expert.to(self.device)
            act_expert = act_expert.to(self.device)
            kl_div = -self.policy.compute_log_prob(obs_expert, act_expert).mean()
            policy_loss = regularization_weight * kl_div + ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        self._sync_weight(self.critic_target, self.critic, soft_target_tau=self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def get_policy(self):
        return self.policy


class EnsembleLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))
        torch.nn.init.trunc_normal_(self.weight, std=1 / (2 * in_features ** 0.5))
        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)
        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes


class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local',
                 with_reward=is_with_reward):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size
        self.activation = Swish()
        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)
        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)
        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * torch.tensor(1), requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * torch.tensor(-5), requires_grad=True))

    def forward(self, obs_action):
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        if self.mode == 'local':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)


class EnsembleReward(torch.nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=1, mode='local',
                 use_restr_for_output=True, use_dropout=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = False
        self.ensemble_size = ensemble_size
        self.use_restr = use_restr_for_output
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        self.activation = Swish()
        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)
        self.output_layer = EnsembleLinear(hidden_features, 1, ensemble_size)
        if self.use_restr:
            self.restr = torch.nn.Tanh()

    def forward(self, obs_action):
        output = obs_action
        if self.use_dropout:
            for layer in self.backbones:
                output = self.activation(self.dropout(layer(output)))
        else:
            for layer in self.backbones:
                output = self.activation(layer(output))

        output = self.output_layer(output)
        if self.use_restr:
            output = self.restr(output)
        return output

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)


class MOPOBuffer:

    def __init__(self, buffer_size):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data):
        batch_data.to_torch(device='cpu')
        if self.data is None:
            self.data = batch_data
        else:
            self.data.cat_(batch_data)

        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size:]

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=batch_size)
        return self.data[indexes]


class AlgoTrainer(BaseAlgo):

    def __init__(self, algo_inst, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        self.transition = algo_inst['transition']['net']
        self.transition_optim = algo_inst['transition']['opt']
        self.selected_transitions = None
        self.reward_function = algo_inst['reward_function']['net']
        self.reward_optim = algo_inst['reward_function']['opt']
        if self.args['use_data_replay']:
            self.reward_buffer = MOPOBuffer(int(self.args['reward_buffer_size']))
        self.use_dropout = self.args['use_dropout']
        self.sac_agent = algo_inst['sac']['sac']
        self.args_sac = algo_inst['args_sac']['args_sac']
        self.log_alpha = algo_inst['log_alpha']['net']
        self.log_alpha_optim = algo_inst['log_alpha']['opt']
        self.all_result_env = []
        self.task_name = args['diverse_data'][5:]
        self.env = gym.make(args['diverse_data'][5:])
        self.max_uncertainty = 0
        self.avg_uncertainty = torch.tensor(0)
        self.z = 0
        self.iter = 0
        self.algo_inst = algo_inst
        self.rew_env = []
        self.rew_expert = 1e6
        self.device = args['device']

    def train(self, train_buffer=None, val_buffer=None, expert_buffer=None):
        file = open('./result/' + self.args['diverse_data'][5:] + '.txt', mode='a')
        whole_buffer = Batch.cat([train_buffer, expert_buffer])
        whole_buffer = SampleBatch(whole_buffer)
        transition = self.train_transition(whole_buffer)
        transition.requires_grad_(False)
        train_buffer, expert_buffer, self.max_uncertainty, self.avg_uncertainty, num_data_large_uncertainty, \
        num_data_small_uncertainty, pos_weight, neg_weigh = self.embed_weight(transition, whole_buffer, train_buffer, expert_buffer)
        for _ in range(self.args['max_iter']):
            self.iter += 1
            model_buffer = self.train_policy(whole_buffer, val_buffer, transition, expert_buffer)
            self.train_reward(transition, train_buffer, expert_buffer, model_buffer)
            print(f'\nAverage return: {self.rew_env}\n')
            file.write(str(self.all_result_env))
            file.write("\n")
        return

    def get_policy(self):
        return self.sac_agent.get_policy()

    def embed_weight(self, transition, whole_buffer, train_buffer, expert_buffer):
        uncertainty, max_uncertainty = comp_uncertainty_batch(self.args, whole_buffer, transition)
        avg_uncertainty = torch.mean(uncertainty)
        weight_train, weight_expert, num_data_large_uncertainty, num_data_small_uncertainty, pos_weight, neg_weigh = comp_weight(
            self.args, train_buffer, expert_buffer, transition, avg_uncertainty,
            max_uncertainty)
        train_buffer = SampleBatch(merge_data_weight(train_buffer, weight_train))
        expert_buffer = SampleBatch(merge_data_weight(expert_buffer, weight_expert))
        return train_buffer, expert_buffer, max_uncertainty, avg_uncertainty, num_data_large_uncertainty, num_data_small_uncertainty, pos_weight, neg_weigh

    def train_transition(self, buffer):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['transition_batch_size']
        val_losses = [float('inf')] * self.transition.ensemble_size
        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
            info = {'iter': epoch, 'loss': np.mean(val_losses)}
            pbar = tqdm(range(int(np.ceil(idxs.shape[-1] / batch_size))), desc='Train dyna', postfix=info,
                        ncols=150)
            for batch_num in pbar:
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                self._train_transition(self.transition, batch, self.transition_optim)
            new_val_losses = self._eval_transition(self.transition, valdata, is_with_reward)
            change = False
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                if new_loss < old_loss:
                    change = True
                    val_losses[i] = new_loss
            if change:
                cnt = 0
            else:
                cnt += 1
            if cnt >= 5:
                break
        val_losses = self._eval_transition(self.transition, valdata, is_with_reward)
        indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
        self.transition.set_select(indexes)
        return self.transition

    def train_policy(self, train_buffer, _, transition, expert_buffer):
        if self.use_dropout:
            self.reward_function.eval()
        real_batch_size = int(self.args['policy_batch_size'] * self.args['real_data_ratio'])
        model_batch_size = self.args['policy_batch_size'] - real_batch_size
        model_buffer = MOPOBuffer(self.args['buffer_size'])
        with torch.no_grad():
            obs_real = torch.tensor(train_buffer['obs'])
            obs_real = torch.tensor(obs_real, device=self.device)
            act_real = torch.tensor(train_buffer['act'])
            act_real = torch.tensor(act_real, device=self.device)
            obs_act_real = torch.cat([obs_real, act_real], dim=-1)
            rew_real = torch.squeeze(self.reward_function(obs_act_real), dim=0)
            train_buffer['rew'] = rew_real.cpu()
            obs_real = torch.tensor(expert_buffer['obs'])
            obs_real = torch.tensor(obs_real, device=self.device)
            act_real = torch.tensor(expert_buffer['act'])
            act_real = torch.tensor(act_real, device=self.device)
            obs_act_real = torch.cat([obs_real, act_real], dim=-1)
            rew_real = torch.squeeze(self.reward_function(obs_act_real), dim=0)
            expert_buffer['rew'] = rew_real.cpu()
        info = {'cri_loss': 'none',
                'act_loss': 'none',
                'return': 'none'}
        pbar = tqdm(range(self.args['init_step'] if self.iter <= self.args['late_start'] else self.args['max_epoch']),
                    desc='Update policy (' + str(self.iter) + '/' + str(self.args['max_iter']) + ')', postfix=info,
                    ncols=150)
        res = {}
        for epoch in pbar:
            model_buffer_current_policy = MOPOBuffer(self.args['horizon'] * self.args['data_collection_per_epoch'])
            with torch.no_grad():
                obs = expert_buffer.sample(int(self.args['data_collection_per_epoch']))['obs']
                obs = torch.tensor(obs, device=self.device)
                for t in range(self.args['horizon']):
                    action = self.sac_agent.select_action(obs)
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = transition(obs_action)
                    next_obses = next_obs_dists.sample()
                    if is_with_reward:
                        next_obses = next_obses[:, :, :-1]
                    else:
                        next_obses = next_obses
                    model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]
                    with torch.no_grad():
                        reward = torch.squeeze(self.reward_function(obs_action), dim=0)
                    dones = torch.zeros_like(reward)
                    batch_data = Batch({
                        "obs": obs.cpu(),
                        "act": action.cpu(),
                        "rew": reward.cpu(),
                        "done": dones.cpu(),
                        "obs_next": next_obs.cpu(),
                    })
                    if self.args['horizon'] == 1:
                        model_buffer.put(batch_data)
                    elif t > 0:
                        model_buffer.put(batch_data)
                    else:
                        pass
                    model_buffer_current_policy.put(batch_data)
                    obs = next_obs
            critic_loss = 0
            actor_loss = 0
            for t in range(self.args['steps_per_epoch']):
                if real_batch_size == 0:
                    batch = model_buffer.sample(model_batch_size)
                else:
                    if self.args['use_expert_data']:
                        batch = expert_buffer.sample(real_batch_size)
                    else:
                        batch = train_buffer.sample(real_batch_size)
                    model_batch = model_buffer.sample(model_batch_size)
                    batch.cat_(model_batch)
                batch.to_torch(device=self.device)
                if self.args['use_clare_regularization']:
                    if self.args['use_expert_behavior']:
                        batch_expert = expert_buffer.sample(int(self.args['policy_batch_size']))
                    else:
                        batch_expert = train_buffer.sample(int(self.args['policy_batch_size']))
                    batch_expert.to_torch(device=self.device)
                    critic_1_loss, critic_2_loss, actor_loss, ent_loss = self.sac_agent.update_parameters(batch, batch_expert, self.args['regularization_weight'])
                else:
                    critic_1_loss, critic_2_loss, actor_loss, ent_loss = self.sac_agent.update_parameters(batch)
                critic_loss = critic_1_loss + critic_2_loss
            if epoch == 0 or (epoch + 1) % self.args['output'] == 0:
                avg_reward = 0.
                episodes = 10
                for _ in range(episodes):
                    state = self.env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = self.sac_agent.get_action(state, evaluate=True)
                        next_state, reward, done, _ = self.env.step(action)
                        episode_reward += reward
                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes
                res = OrderedDict()
                res["Reward_Mean_Env"] = avg_reward
                res['normalized average uncertainty'] = self.avg_uncertainty.item()
                batch_simu = model_buffer_current_policy.sample(int(self.args['data_collection_per_epoch']))
                res['reward (model)'] = batch_simu['rew'].mean().item()
                batch_real = expert_buffer.sample(int(self.args['data_collection_per_epoch']))
                res['reward (real)'] = batch_real['rew'].mean().item()
                res['critic loss'] = float(critic_loss)
                res['actor loss'] = float(actor_loss)
                self.all_result_env.append(res["Reward_Mean_Env"])
                self.rew_env.append(round(res["Reward_Mean_Env"], 1))
                info = {'cri_loss': res['critic loss'],
                        'act_loss': res['actor loss'],
                        'return': res['Reward_Mean_Env']}
                pbar.set_postfix(info)
            else:
                batch_simu = model_buffer_current_policy.sample(int(self.args['data_collection_per_epoch']))
                res['reward (model)'] = batch_simu['rew'].mean().item()
                batch_real = expert_buffer.sample(int(self.args['data_collection_per_epoch']))
                res['reward (real)'] = batch_real['rew'].mean().item()
                res['critic loss'] = float(critic_loss)
                res['actor loss'] = float(actor_loss)
                info = {'cri_loss': res['critic loss'],
                        'act_loss': res['actor loss'],
                        'return': res['Reward_Mean_Env']}
                pbar.set_postfix(info)
        if self.use_dropout:
            self.reward_function.train()
        return model_buffer

    def train_reward(self, _, train_buffer, expert_buffer, model_buffer):
        whole_buffer = SampleBatch(Batch.cat([train_buffer, expert_buffer]))
        self.z = self.compute_z(whole_buffer)
        if self.args['use_data_replay']:
            self.reward_buffer.put(model_buffer.sample(int(self.args['sample_data'])))
            model_buffer = self.reward_buffer
        info = {'reward loss': 'inf'}
        pbar = tqdm(range(self.args['steps_per_reward_update']),
                    desc='Update reward (' + str(self.iter) + '/' + str(self.args['max_iter']) + ')',
                    postfix=info, ncols=150)
        for _ in pbar:
            model_batch = model_buffer.sample(int(self.args['batch_size_reward']))
            whole_batch = whole_buffer.sample(int(self.args['batch_size_reward']))
            expert_batch = expert_buffer.sample(int(self.args['batch_size_reward']))
            loss = self._train_reward(self.reward_function, self.reward_optim, model_batch, whole_batch, expert_batch)
            info = {'rew_loss': round(float(loss), 3)}
            pbar.set_postfix(info)

    def _train_reward(self, reward_function, optim, model_batch, whole_batch, expert_batch):
        """Update reward function one step."""
        model_batch.to_torch(device=self.device)
        whole_batch.to_torch(device=self.device)
        expert_batch.to_torch(device=self.device)
        model_batch_obs_act = torch.cat([model_batch['obs'], model_batch['act']], dim=-1)
        model_reward = torch.mean(reward_function(model_batch_obs_act))
        whole_batch_obs_act = torch.cat([whole_batch['obs'], whole_batch['act']], dim=-1)
        whole_reward = torch.mean(reward_function(whole_batch_obs_act) * whole_batch['weight'])
        expert_batch_obs_act = torch.cat([expert_batch['obs'], expert_batch['act']], dim=-1)
        expert_reward = torch.mean(reward_function(expert_batch_obs_act))
        num_data = model_batch.shape[0] + whole_batch.shape[0]
        avg_reward_norm = (torch.sum(torch.squeeze(reward_function(model_batch_obs_act)) ** 2)
                           + torch.sum(torch.squeeze(reward_function(whole_batch_obs_act)) ** 2)) / num_data
        regularizer = avg_reward_norm
        if self.args['use_regularizer']:
            loss = self.z * model_reward - whole_reward - expert_reward \
                   + self.args['regularizer_weight'] * self.z * regularizer
        else:
            loss = self.z * model_reward - whole_reward - expert_reward
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    @staticmethod
    def compute_z(whole_buffer):
        return 1 + np.mean(whole_buffer['weight'])

    @staticmethod
    def _select_best_indexes(metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_transition(self, transition, dataset, optim):
        dataset.to_torch(device=self.device)
        dist = transition(torch.cat([dataset['obs'], dataset['act']], dim=-1))
        if is_with_reward:
            loss = - dist.log_prob(torch.cat([dataset['obs_next'], dataset['rew']], dim=-1))
        else:
            loss = - dist.log_prob(dataset['obs_next'])
        loss = loss.mean()
        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    def _eval_transition(self, transition, valdata, with_reward):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
            if with_reward:
                loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1, 2))
            else:
                loss = ((dist.mean - valdata['obs_next']) ** 2).mean(dim=(1, 2))
            return list(loss.cpu().numpy())
