import numpy as np
import pickle as pkl
import argparse
import subprocess
import random
import sys
import ipdb
import json
import warnings
from pathlib import Path

import optuna.integration.lightgbm as lgb
from box import Box

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

    model_path = \
        Path(config.model_dir) / config.model_name
    if not model_path.is_dir():
        model_path.mkdir(parents=True)

    logger.info(f'[+] Model loaded from {model_path / "model"}')
    with open(model_path / 'model', 'rb') as handle:
        model = pkl.load(handle)
    strategy = getattr(strategies, config.strategy)(
        config=config,
        model=model
    )
    
    logger.info('[*] Start Testing...')
    output, test_log = strategy.test()
    output.to_csv(exp_path / "prediction.csv")
    
    with open(exp_path / "log.json", 'w') as f:
        log = {'Testing Log': test_log}
        f.write(json.dumps(log, indent=4) + "\n")

    logger.info(f'[+] Experiment log dumped at {exp_path / "log.json"}')

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
