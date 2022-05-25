import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch

def load_d4rl_buffer(task):
    print(task)
    env = gym.make(task[5:]) # task[0:4] = 'd4rl-'
    dataset = d4rl.qlearning_dataset(env)
    # print(type(dataset['observations']))
    num_index=[]
    nums_1 = 0
    for i in range(dataset['actions'].shape[0]):
        if(dataset['actions'][i].min()==-1 or dataset['actions'][i].max()==1):
            num_index.append(i)
            nums_1+=1
    if len(num_index) != 0:
        dataset['observations']=np.delete(dataset['observations'], num_index, axis=0)
        dataset['actions']=np.delete(dataset['actions'], num_index, axis=0)
        dataset['next_observations']=np.delete(dataset['next_observations'], num_index, axis=0)
        dataset['rewards']=np.delete(dataset['rewards'], num_index, axis=0)
        dataset['terminals']=np.delete(dataset['terminals'], num_index, axis=0)
    # print("nums_1",nums_1)
    # print(dataset['rewards'])
    buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )
    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer
