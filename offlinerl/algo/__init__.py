from loguru import logger
import warnings

warnings.filterwarnings('ignore')


from offlinerl.config.algo import  clare_config
from offlinerl.utils.config import parse_config
from offlinerl.algo.modelbase import clare

algo_dict = {'clare' : {"algo" : clare, "config" : clare_config}}

def algo_select(command_args, algo_config_module=None):
    algo_name = 'clare'
    logger.info('Use CLARE!')
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]  # 'algo' is the package of the selected algorithm
    
    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)  # Read the configuration of the selected algorithm
    algo_config.update(command_args)  # Cover default configurations by input arguments
    
    algo_init = algo.algo_init  # 'algo_init' is the initialization function
    algo_trainer = algo.AlgoTrainer  # 'algo_trainer' is a class
    
    return algo_init, algo_trainer, algo_config
    
    