# Conservative Model-Based Reward Learning (CLARE)


## Installation

### Download mujoco210
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco

# Set environment path in ~/.bashrc
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
# export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
# export MUJOCO_KEY_PATH=$MUJOCO_KEY_PATH:/root/.mujoco
source ~/.bashrc
```

### Install D4RL
```
git clone https://github.com/rail-berkeley/d4rl
cd d4rl
# Comment out dm-control and mujoco-py in setup.py
pip install -e .
```

### Install NeoRL
```
git clone https://agit.ai/Polixir/neorl.git
cd neorl
pip install -e .
```

### Install CLARE
```
pip install -e .
```


## Train & Evaluate CLARE

To train the policy and reward function, run this command:

```train
python train_clare.py --diverse_data='d4rl-halfcheetah-medium-v2' --expert_data='d4rl-halfcheetah-expert-v2' --u=0.5
```
The tasks and datasets we train on can be selected through `diverse_data` and `expert_data`. The conservatism level is controlled by the value of `u` (between 0 and 1). The values of `num_diverse_data` and `num_expert_data` can be set to control the number of state-action pairs. The average returns are saved in `result` directory.



