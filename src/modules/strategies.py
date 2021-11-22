from pathlib import Path
from tqdm import tqdm

import numpy as np
import pickle as pkl
import modules.utils as utils
import modules.preprocessors as preprocessors
import modules.models as models
import modules.pricers as pricers
from modules.logger import create_logger

from sklearn.model_selection import KFold

class Strategy:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.
    """

    def __init__(self, config):
        self.config = config
        self.train_data, self.valid_data = {}, {}
        data_dir = Path(config.data_dir)
        for cat in ['X', 'Y']:
            with open(data_dir / f"{cat}_train.pkl" , 'rb') as dataFile:
                self.train_data[cat] = pkl.load(dataFile)
            with open(data_dir / f"{cat}_valid.pkl" , 'rb') as dataFile:
                self.valid_data[cat] = pkl.load(dataFile)
            
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

        keys = ['FOLD', 'NLL LOSS', 'ACC']
        eval_bar.write(utils.getTimeStr() + ''.join(f"{key:>10}" for key in keys))
        
        log = []
        for i, (train_index, test_index) in enumerate(eval_bar):
            big_fold = {
                'X': self.train_data['X'][train_index], 
                'Y': self.train_data['Y'][train_index]
            }
            small_fold = {
                'X': self.train_data['X'][test_index], 
                'Y': self.train_data['Y'][test_index]
            }

            self.model.fit(big_fold)
            proba = self.model.predict_proba(small_fold)
            log_loss = utils.NLLLoss(proba, small_fold['Y'])
            acc = utils.Accuracy(proba, small_fold['Y'])

            eval_bar.write(utils.getTimeStr() + ''.join(f"{key:>10.3f}" for key in [i, log_loss, acc]))
            log.append({f"Fold {i}": {'Negative Log Likelihood': log_loss, \
                                      'Accuracy': acc}})

        self.logger.info(f'[*] Training with all data')
        self.model.fit(self.train_data)
        
        return log

    def valid(self):
        provided_revenue = 0
        for (x, y) in zip(self.valid_data["X"], self.valid_data["Y"]):
            if y == 1:
                provided_revenue += x[-2]
            elif y == 2:
                provided_revenue += x[-1]
        provided_revenue /= len(self.valid_data["X"])
        self.logger.info(f'[-] Average Revenue Provided: {provided_revenue:2.3f}')

        self.pricer = getattr(pricers, self.config.valid.pricer)(
            param=self.config.valid.param
        )

        expected_revenue, predicted_prices = self.pricer.run(self.model, self.valid_data)            
        
        self.logger.info(f'[-] Expected Average Revenue: {expected_revenue}')

        return {'Average Revenue Provided': provided_revenue, 
                'Expected Average Revenue': expected_revenue}

    def test(self, x):
        pass

    