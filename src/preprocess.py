import numpy as np
import pandas as pd
import argparse
import random
import sys
import ipdb

import pickle as pkl
from box import Box
from pathlib import Path
from sklearn.model_selection import train_test_split

import modules.utils as utils
import modules.preprocessors as preprocessors
from modules.logger import create_logger

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

    logger = create_logger(name="PREPROCESS")
    logger.info(f'[-] Config loaded from {config_path}')

    # Set seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Read Data
    data_dir = Path(config.data_dir)
    data = {}
    logger.info(f'[-] Training data loaded from {data_dir}')    
    for name, fileName in config.data.train.items():
        data[name] = utils.openData(data_dir / fileName)
    n_training_data = data['covariate'].shape[0]

    # Run Preprocessor
    preprocessor = getattr(preprocessors, config.preprocess.preprocessor)(
        param=config.preprocess.param
    )
    preprocessed_data = preprocessor.run(data)

    # Get Random Sample Indices
    output = train_test_split(preprocessed_data['X'], preprocessed_data['Y'], \
                              test_size=config.preprocess.valid_size, \
                              random_state=config.random_seed)

    # Write training data to pickle
    for name, split in zip(['X_train', 'X_valid', 'Y_train', 'Y_valid'], output):
        with open(data_dir / f"{name}.pkl", "wb") as output_file:
            pkl.dump(split, output_file)
        logger.info(f'[+] Dumped {name}.pkl at {data_dir / f"{name}.pkl"}')

    logger.info(f'[-] Teating data loaded from {data_dir}')
    for name, fileName in config.data.test.items():
        test_data = utils.openData(data_dir / fileName)
        test_data.columns = data[name].columns
        data[name] = data[name].append(test_data)

    preprocessed_data = preprocessor.run(data, mode='test')
    
    # Write testing data to pickle
    with open(data_dir / f"X_test.pkl", "wb") as output_file:
        pkl.dump(preprocessed_data['X'][n_training_data:], output_file)
    logger.info(f'[+] Dumped X_test.pkl at {data_dir / f"X_test.pkl"}')
    
if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
