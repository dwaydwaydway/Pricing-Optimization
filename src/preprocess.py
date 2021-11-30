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
import modules.vectorRegressors as vectorRegressors
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

    logger.info(f'[-] Teating data loaded from {data_dir}')
    for name, fileName in config.data.test.items():
        test_data = utils.openData(data_dir / fileName)
        test_data.columns = data[name].columns
        data[name] = data[name].append(test_data)

    # Run Vector Regressor
    regressor = getattr(vectorRegressors, config.preprocess.regressor)(
        param=config.preprocess.param
    )
    regressor.run(data)

    # Include item embedding data
    if config.preprocess.vectors == 'concat':
        n_data = len(data['noisy_embedding'])
        item0embedding, item1embedding = pd.DataFrame([data['item0embedding']]), pd.DataFrame([data['item1embedding']])
        item0embedding, item1embedding = pd.concat([item0embedding]*n_data), pd.concat([item1embedding]*n_data)
        # Set Column Name for item/user vectors
        item0embedding.columns = [f'item0_vector_{i}' for i in range(10)]
        item1embedding.columns = [f'item1_vector_{i}' for i in range(10)]
        # Concatenate All Features
        X = pd.concat([data['noisy_embedding'].reset_index(drop=True), \
                        data['covariate'].reset_index(drop=True), 
                        item0embedding.reset_index(drop=True), \
                        item1embedding.reset_index(drop=True)], axis=1)

    elif self.vectors == 'dot':
        item0embedding, item1embedding = pd.DataFrame(data['item0embedding']), pd.DataFrame(data['item1embedding'])
        data['user_dot_item0'] = data['noisy_embedding'].dot(item0embedding)
        data['user_dot_item1'] = data['noisy_embedding'].dot(item1embedding)
        X = pd.concat([data['covariate'].reset_index(drop=True), \
                        data['user_dot_item0'].reset_index(drop=True), \
                        data['user_dot_item1'].reset_index(drop=True)], axis=1)
    
    X_train, X_test = X.iloc[:n_training_data], X.iloc[n_training_data:]
    
    # Get the training label
    prices = data['prices_decisions'].drop(columns=['user_index', 'item_bought'])
    X_train = pd.concat([X_train, prices.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(np.zeros((X_test.shape[0], 2))).reset_index(drop=True)], axis=1)
    Y_train = [row+1 for row in data['prices_decisions']['item_bought']]

    X_train, X_test, Y_train = X_train.to_numpy(), X_test.to_numpy(), np.array(Y_train).reshape(-1)

    # Write training/testing data to pickle
    with open(data_dir / "X_train.pkl", "wb") as output_file:
        pkl.dump(X_train, output_file)
    logger.info(f'[+] Dumped X_train.pkl ({np.array(X_train).shape}) at {data_dir / "X_train.pkl"}')
    with open(data_dir / "Y_train.pkl", "wb") as output_file:
        pkl.dump(Y_train, output_file)
    logger.info(f'[+] Dumped Y_train.pkl ({np.array(Y_train).shape}) at {data_dir / "Y_train.pkl"}')
    with open(data_dir / f"X_test.pkl", "wb") as output_file:
        pkl.dump(X_test, output_file)
    logger.info(f'[+] Dumped X_test.pkl ({np.array(X_test).shape}) at {data_dir / f"X_test.pkl"}')
    
if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
