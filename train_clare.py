import fire
from offlinerl.algo import algo_select
from offlinerl.evaluation import OnlineCallBackFunction
from offlinerl.data.d4rl import load_d4rl_buffer
from offlinerl.utils.data import SampleBatch


def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    if 'random' in algo_config["diverse_data"]:
        print("Diverse data type: random\n")

    if 'expert' in algo_config["diverse_data"]:
        algo_config["only_expert"] = True
        print("Diverse data type: expert\n")
    else:
        algo_config["only_expert"] = False

    if 'medium' in algo_config["diverse_data"]:
        print("Diverse data type: medium\n")
        if algo_config["num_expert_data"] < 1e4 and algo_config["with_beta"]:
            algo_config["use_expert_behavior"] = False
        
        if 'halfcheetah' in algo_config["diverse_data"]:
            algo_config["u"] = 0.1
            # algo_config["use_expert_behavior"] = False
        
        if algo_config["num_expert_data"] <= 2e4 and algo_config["with_beta"] and 'halfcheetah' in algo_config["diverse_data"]:
            # algo_config["u"] = 0.1
            algo_config["use_expert_behavior"] = False

    # Get diverse data
    train_buffer = load_d4rl_buffer(algo_config["diverse_data"])
    # Get expert data
    expert_buffer = load_d4rl_buffer(algo_config["expert_data"])

    # Instantiate the algorithm
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = OnlineCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=None, task=algo_config["diverse_data"])

    print(f"u: {algo_config['u']}")

    if algo_config['use_limited_data']:
        if algo_config['only_expert']:
            expert_buffer = SampleBatch(expert_buffer.sample(algo_config['num_expert_data'] + algo_config['num_diverse_data']))
            # Train buffer uses expert dataset
            algo_trainer.train(train_buffer=expert_buffer, val_buffer=None, expert_buffer=expert_buffer)
        else:
            train_buffer = SampleBatch(train_buffer.sample(algo_config['num_diverse_data']))
            expert_buffer = SampleBatch(expert_buffer.sample(algo_config['num_expert_data']))
            algo_trainer.train(train_buffer=train_buffer, val_buffer=None, expert_buffer=expert_buffer)
    else:
        train_buffer = SampleBatch(train_buffer)
        expert_buffer = SampleBatch(expert_buffer)
        if algo_config['only_expert']:
            # Train buffer uses expert dataset
            algo_trainer.train(train_buffer=expert_buffer, val_buffer=None, expert_buffer=expert_buffer)
        else:
            algo_trainer.train(train_buffer=train_buffer, val_buffer=None, expert_buffer=expert_buffer)

    print("Finish successfully!")

if __name__ == "__main__":

    fire.Fire(run_algo)

    # tasks = [
    #     ['walker2d-medium-v2', 'walker2d-expert-v2'],          # dynamics_loss: (low) 0.866
    #     ['hopper-medium-v2', 'hopper-expert-v2'],              # dynamics_loss: (low) 1.16
    #     ['halfcheetah-medium-v2', 'halfcheetah-expert-v2'],    # dynamics_loss: (low) 0.219
    #     ['ant-medium-v2', 'ant-expert-v2']
    # ]
    
    # for num in [True, False]:
    #     for task in tasks:
    #         print(f'\r{task} starts!')
    #         run_algo(diverse_data='d4rl-' + task[0],
    #                  expert_data='d4rl-' + task[1],
    #                  with_beta = num
    #                 )


