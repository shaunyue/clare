import torch
import numpy as np
from offlinerl.utils.exp import select_free_cuda


device = 'cuda' + ":" + str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
seed = np.random.randint(1, 2022)

use_limited_data = True
only_expert = False
diverse_data = 'd4rl-halfcheetah-medium-v2'
expert_data = 'd4rl-halfcheetah-expert-v2'
num_diverse_data = 10000
num_expert_data = 10000

# Expert data
# 'd4rl-ant-expert-v2'
# 'd4rl-halfcheetah-expert-v2'
# 'd4rl-walker2d-expert-v2'
# 'd4rl-hopper-expert-v2'
# Medium data
# 'd4rl-ant-medium-v2'
# 'd4rl-halfcheetah-medium-v2'
# 'd4rl-walker2d-medium-v2'
# 'd4rl-hopper-medium-v2'
# Random data
# 'd4rl-ant-random-v2'
# 'd4rl-halfcheetah-random-v2'
# 'd4rl-walker2d-random-v2'
# 'd4rl-hopper-random-v2'

# Reward
reward_init_num = 1
reward_select_num = 1
reward_layers = 4
hidden_layer_size_reward = 256
use_restr_for_output = False
use_regularizer = True
regularizer_weight = 1
use_dropout = False
reward_lr = 5e-5
reward_buffer_size = 1.2e6
batch_size_reward = 5000
use_data_replay = True
sample_data = 2.5e4

# Policy
horizon = 5
hidden_size = 256
policy_batch_size = 256
data_collection_per_epoch = 5000
buffer_size = 2.5e4

# SAC
sac_target_update_interval = 1
sac_updates_per_step = 1
sac_batch_size = 256
sac_automatic_entropy_tuning = True
sac_alpha = 0.2
sac_eval = True
sac_gamma = 0.99
sac_tau = 0.005
sac_lr = 3e-4

# Train CLARE
max_iter = 10
max_epoch = 500
steps_per_epoch = 20
steps_per_reward_update = 5
output = 100
init_step = 20
late_start = 1

use_clare_regularization = True
regularization_weight = 0.25
with_beta = True
u = 0.5
uncertainty_mode = 'aleatoric'  # Or 'disagreement'

# Transition
transition_batch_size = 256
hidden_layer_size = 256
hidden_layers = 2
transition_layers = 4
transition_init_num = 7
transition_select_num = 5
transition_lr = 1e-3

# Do not change
algo_name = 'clare'
exp_name = 'experiment'
is_avg_based = False
beta = 0
use_expert_data = False
use_expert_behavior = True
real_data_ratio = 0.0