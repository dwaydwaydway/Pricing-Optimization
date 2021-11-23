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
    logger.info(f'[-] Tranning Model {config.model_name}')

    # Set seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Make a dictionary for this experiment
    model_path = \
        Path(config.model_dir) / config.model_name
    if not model_path.is_dir():
        model_path.mkdir(parents=True)

    logger.info('[*] Start training...')
    strategy = getattr(strategies, config.strategy)(
        config=config,
    )    
    model, train_log = strategy.train()        
    with open(model_path / 'model', 'wb') as handle:
        pkl.dump(model, handle)
    logger.info(f'[+] Model dumped at {model_path / "model"}')

    logger.info('[*] Start Validating...')
    valid_log = strategy.valid()

    with open(model_path / f"log.json", 'w') as f:
        log = {'Traning Log': train_log, 'Validation Log': valid_log}
        f.write(json.dumps(log, indent=4) + "\n")
    logger.info(f'[+] Training/Validation Log dumped at {model_path / f"log.json"}')

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
