import time
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import modules.utils as utils
import modules.preprocessors as preprocessors
import modules.models as models
import modules.pricers as pricers
from modules.logger import create_logger


class Strategy:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.
    """

    def __init__(self, config):
        self.config = config
        self.train_data, self.valid_data, self.test_data = {}, {}, {}
        data_dir = Path(config.data_dir)
        for cat in ['X', 'Y']:
            with open(data_dir / f"{cat}_train.pkl" , 'rb') as dataFile:
                self.train_data[cat] = pkl.load(dataFile)
            with open(data_dir / f"{cat}_valid.pkl" , 'rb') as dataFile:
                self.valid_data[cat] = pkl.load(dataFile)
        with open(data_dir / f"X_test.pkl" , 'rb') as dataFile:
            self.test_data['X'] = pkl.load(dataFile)
            
        self.model = getattr(models, self.config.train.model)(
            param=config.train.param
        )
        self.logger = create_logger(name="STRATEGY")

    def train(self, *args, **kwargs):
        """
        Higher-order map.
        .. image:: figs/Ops/maplist.png
        See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_
        Args:
            fn (one-arg function): Function from one value to one value.
        Returns:
            function : A function that takes a list, applies `fn` to each element, and returns a
            new list
        """
        raise NotImplementedError(f"train function not implemented for {self.__class__.__name__}")
    
    def test(self, *args, **kwargs):
        raise NotImplementedError(f"test function not implemented for {self.__class__.__name__}")

class BaseStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)

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
        for i, (train_index, test_index) in enumerate(eval_bar):
            big_fold = {
                'X': self.train_data['X'][train_index], 
                'Y': self.train_data['Y'][train_index]
            }
            small_fold = {
                'X': self.train_data['X'][test_index], 
                'Y': self.train_data['Y'][test_index]
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
            log[f"Fold {i}"] = {'Train':
                                    {'Negative Log Likelihood': train_nllLoss, \
                                    'Accuracy': train_acc},
                               'Valid':
                                    {'Negative Log Likelihood': valid_nllLoss, \
                                    'Accuracy': valid_acc}
                            }
        end_time = time.time()

        self.logger.info(f'[*] Training with all data')
        self.model.fit(self.train_data, self.config.train.fit_params)

        log['Time'] = end_time - start_time

        return log

    def valid(self):
        provided_revenue = 0
        for (x, y) in zip(self.valid_data["X"], self.valid_data["Y"]):
            if y == 1:
                provided_revenue += x[-2]
            elif y == 2:
                provided_revenue += x[-1]
        
        self.logger.info(f'[-] Average Revenue Provided: {provided_revenue / len(self.valid_data["X"]):2.3f}')

        self.pricer = getattr(pricers, self.config.valid.pricer)(
            param=self.config.valid.param
        )
        start_time = time.time()
        expected_revenue, predicted_prices = self.pricer.run(self.model, self.valid_data)            
        end_time = time.time()

        self.logger.info(f'[-] [VALID] Expected Average Revenue: {sum(expected_revenue) / len(expected_revenue):2.3f}')

        return {'Average Revenue Provided': provided_revenue / len(self.valid_data["X"]), 
                'Expected Average Revenue': sum(expected_revenue) / len(expected_revenue), 
                'Time': end_time - start_time}

    def test(self):
        start_time = time.time()
        expected_revenue, predicted_prices = self.pricer.run(self.model, self.test_data)
        end_time = time.time()

        output = {'user_index': list(range(14000, 14000+2912)), 
                  'price_item_0': np.array([price[0] for price in predicted_prices], dtype=np.float32), 
                  'price_item_1': np.array([price[1] for price in predicted_prices], dtype=np.float32), 
                  'expected_revenue': np.array(expected_revenue, dtype=np.float32)
        }

        self.logger.info(f'[-] [TEST] Expected Average Revenue: {sum(expected_revenue) / len(expected_revenue):2.3f}')

        output = pd.DataFrame(output)
        return output, {'Expected Average Revenue': sum(expected_revenue) / len(expected_revenue), 
                        'Time': end_time - start_time}

    