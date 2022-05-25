# Conservative Model-Based Reward Learning (CLARE)

---
## Requirement & Installation

####Installation
- Create a conda environment and install mujoco-py and mujoco210
  - Mujoco-py can be installed with the following command: pip install -U 'mujoco-py<2.2,>=2.1'
- Install d4rl, which needs to be downloaded from https://github.com/rail-berkeley/d4rl
  - Note that during installing d4rl, dm_control don't need to be installed, while mjrl requires installing additionally (with command: pip install git+Https://github.com/aravindr93/mjrl@master#egg=mjrl). These two lines of commands need to be commented out in the setup.py
- Install neorl
- Install other dependencies: pip install -r requirements.txt

####Requirement
- pytorch~=1.11.0
- mujoco-py<2.2,>=2.1
- torch~=1.11.0
- tianshou~=0.4.8
- setuptools~=61.2.0
- loguru~=0.6.0
- numpy~=1.21.6
- gym~=0.23.1
- sklearn~=0.0
- scikit-learn~=1.0.2
- tqdm~=4.64.0
- scipy~=1.7.3
- aim~=2.0.27
- ray~=1.12.0
- patchelf

---


## Train & Evaluate CLARE

To train the policy and reward function, run this command:

```train
python train_d4rl.py --diverse_data='d4rl-halfcheetah-medium-v2' --expert_data='d4rl-halfcheetah-expert-v2' --u=0.5
```
The tasks and datasets we train on can be selected through diverse_data and expert_data. The conservatism level is controlled by the value of u (between 0 and 1). Additionally, num_diverse_data and num_expert_data can be set to control the number of state-action pairs we use.

The average returns are saved in `result` directory.


## Results

The performance of CLARE are shown as follows (under 10k tuples):


<table>
	<tr>
	    <th>Dataset Type</th>
	    <th>Environment</th>
      <th>CLARE</th> 
	</tr >
	<tr >
	    <td rowspan="4"><center>expert + random</center>
</td>
	    <td>walker2d</td>
	    <td>3673.8</td>
	</tr>
	<tr>
	    <td>hopper</td>
	    <td>1891.5</td>
	</tr>
	<tr>
	    <td>ant</td>
	    <td>1960.0</td>
	</tr>
	<tr>
	    <td>halfcheetah</td>
	    <td>1113.7</td>
	</tr>
	<tr >
	    <td rowspan="4"><center>expert + medium</center></td>
	    <td>walker2d</td>
	    <td>3813.4</td>
	</tr>
	<tr>
	    <td>hopper</td>
	    <td>2335.0</td>
	</tr>
	<tr>
	    <td>ant</td>
	    <td>3879.4</td>
	</tr>
	<tr>
	    <td>halfcheetah</td>
	    <td>4888.6</td>
	</tr>
	<tr >
	    <td rowspan="4"><center>expert</center></td>
	    <td>walker2d</td>
	    <td>4990.5</td>
	</tr>
	<tr>
	    <td>hopper</td>
	    <td>2604.5</td>
	</tr>
	<tr>
	    <td>ant</td>
	    <td>3940.3</td>
	</tr>
	<tr>
	    <td>halfcheetah</td>
	    <td>5375.1</td>
	</tr>
</table>



