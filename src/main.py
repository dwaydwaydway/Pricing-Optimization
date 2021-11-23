import numpy as np
import argparse
import subprocess
import random
import sys
import ipdb
import json
import warnings

from box import Box
from pathlib import Path

from modules.logger import create_logger
from modules import strategies

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config', dest='config_path',
            default='./config.yaml', type=Path,
            help='the path of config file')
    args = parser.parse_args()
    return vars(args)

def main(config_path):
    config = Box.from_yaml(config_path.open())
    warnings.filterwarnings("ignore", category=UserWarning)

    logger = create_logger(name="MAIN")
    logger.info(f'[-] Config loaded from {config_path}')
    logger.info(f'[-] Running Experiment {config.exp_name}')

    # Set seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Make a dictionary for this experiment
    exp_path = \
        Path(config.exp_dir) / config.exp_name
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True)
    subprocess.call(['cp', config_path, exp_path / "config.yaml"])

    logger.info('[*] Start training...')
    strategy = getattr(strategies, config.strategy)(
            config=config
    )    
    train_log = strategy.train()

    logger.info('[*] Start Validating...')
    valid_log = strategy.valid()

    logger.info('[*] Start Testing...')
    output, test_log = strategy.test()
    output.to_csv(exp_path / "prediction.csv")
    
    with open(exp_path / "log.json", 'w') as f:
        log = {'Training Log': train_log, 'Validation Log': valid_log, 'Testing Log': test_log}
        f.write(json.dumps(log, indent=4) + "\n")

    logger.info(f'[+] Experiment log dumped at {exp_path / "log.json"}')

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
