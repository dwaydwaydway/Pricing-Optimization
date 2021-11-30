import time
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import modules.utils as utils
import modules.models as models
import modules.pricers as pricers
from modules.logger import create_logger


class Strategy:
    def __init__(self, config):
        self.config = config
        self.train_data, self.test_data = {}, {}
        data_dir = Path(config.data_dir)
        for cat in ['X', 'Y']:
            with open(data_dir / f"{cat}_train.pkl" , 'rb') as dataFile:
                self.train_data[cat] = pkl.load(dataFile)
        with open(data_dir / f"X_test.pkl" , 'rb') as dataFile:
            self.test_data['X'] = pkl.load(dataFile)
            
        self.model = getattr(models, self.config.train.model)(
            param=config.train.param
        )
        self.logger = create_logger(name="STRATEGY")

    def train(self, *args, **kwargs):
        raise NotImplementedError(f"train function not implemented for {self.__class__.__name__}")
    
    def test(self, *args, **kwargs):
        raise NotImplementedError(f"test function not implemented for {self.__class__.__name__}")

class BaseStrategy(Strategy):
    def __init__(self, config, model=None):
        super().__init__(config)
        if model:
            self.model = model

    def train(self):
        self.logger.info(f'[*] Performaing {self.config.train.fold} fold Evaluation')
        kf = KFold(n_splits=self.config.train.fold)
        eval_bar = tqdm(
            kf.split(self.train_data['X']),
            total=self.config.train.fold,
            desc='[KFold Cross-Val]',
            leave=False,
            position=0)

        keys = ['SPLIT', 'FOLD', 'NLL', 'ACC']
        eval_bar.write(utils.getTimeStr() + ''.join(f"{key:>10}" for key in keys))
        
        log = {}
        start_time = time.time()
        for i, (train_index, valid_index) in enumerate(eval_bar):
            big_fold = {
                'X': self.train_data['X'][train_index], 
                'Y': self.train_data['Y'][train_index]
            }
            small_fold = {
                'X': self.train_data['X'][valid_index], 
                'Y': self.train_data['Y'][valid_index]
            }
            self.model.fit(big_fold, self.config.train.fit_params)

            train_proba = self.model.predict_proba(big_fold)
            valid_proba = self.model.predict_proba(small_fold)
            train_nllLoss = utils.NLLLoss(train_proba, big_fold['Y'])
            valid_nllLoss = utils.NLLLoss(valid_proba, small_fold['Y'])
            train_acc = utils.Accuracy(train_proba, big_fold['Y'])
            valid_acc = utils.Accuracy(valid_proba, small_fold['Y'])

            eval_bar.write(utils.getTimeStr() + '     TRAIN' + ''.join(f"{key:>10.3f}" for key in [i, train_nllLoss, train_acc]))
            eval_bar.write(utils.getTimeStr() + '     VALID' + ''.join(f"{key:>10.3f}" for key in [i, valid_nllLoss, valid_acc]))
            log[f"Fold {i}"] = {
                'Train':{
                    'Negative Log Likelihood': train_nllLoss, \
                    'Accuracy': train_acc
                },
                'Valid':{
                    'Negative Log Likelihood': valid_nllLoss, \
                    'Accuracy': valid_acc
                }
            }
            self.model.reset()

        end_time = time.time()

        self.logger.info(f'[*] Training with all data')
        self.model.fit(self.train_data, self.config.train.fit_params)

        log['Time'] = end_time - start_time

        return self.model, log

    def test(self):
        self.pricer = getattr(pricers, self.config.test.pricer)(
            param=self.config.test.param
        )
        start_time = time.time()
        predicted_prices, expected_hard_revenue, expected_soft_revenue, expected_penalized_revenue = \
            self.pricer.run(self.model, self.test_data)
        end_time = time.time()

        output = {'user_index': list(range(14001, 14001+2912)), 
                  'price_item_0': np.array([price[0] for price in predicted_prices], dtype=np.float32), 
                  'price_item_1': np.array([price[1] for price in predicted_prices], dtype=np.float32), 
                  'expected_revenue': np.array(expected_soft_revenue, dtype=np.float32)
        }

        self.logger.info(f'[-] [TEST] Expected Average Hard Revenue: {sum(expected_hard_revenue) / len(expected_hard_revenue):2.3f}')
        self.logger.info(f'[-] [TEST] Expected Average Soft Revenue: {sum(expected_soft_revenue) / len(expected_soft_revenue):2.3f}')
        self.logger.info(f'[-] [TEST] Expected Average Penalized Revenue: {sum(expected_penalized_revenue) / len(expected_penalized_revenue):2.3f}')


        output = pd.DataFrame(output, index=None)
        return output, {'Expected Average Hard Revenue': sum(expected_hard_revenue) / len(expected_hard_revenue), 
                        'Expected Average Soft Revenue': sum(expected_soft_revenue) / len(expected_soft_revenue), 
                        'Expected Average Penalized Revenue': sum(expected_penalized_revenue) / len(expected_penalized_revenue), 
                        'Time': end_time - start_time}

    