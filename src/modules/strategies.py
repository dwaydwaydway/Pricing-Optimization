from pathlib import Path
from tqdm import tqdm

import modules.utils as utils
import modules.preprocessors as preprocessors
import modules.models as models

class Strategy:
    def __init__(self, config):
        self.config = config
        self.data = {}
        data_dir = Path(config.data_dir)
        for name, fileName in config.data.items():
            self.data[name] = utils.openData(data_dir / fileName)
        self.preprocessor = getattr(preprocessors, config.preprocess.preprocessor)(
            param=config.preprocess.param
        )
        self.model = getattr(models, self.config.optimize.model)(
            param=config.optimize.param
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(f"train function not implemented for {self.__class__.__name__}")
    
    def test(self, *args, **kwargs):
        raise NotImplementedError(f"test function not implemented for {self.__class__.__name__}")

class BaseStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        self.data = self.preprocessor.run(self.data)

        eval_bar = tqdm(
            utils.get_KFold(self.data, self.config.optimize.fold),
            total=self.config.optimize.fold,
            desc='[KFold Cross-Val]',
            position=0)

        keys = ['Fold', 'item0 CE', 'item1 CE']
        eval_bar.write(utils.getTimeStr() + ''.join(f"{key:>10}" for key in keys))
        
        log = []
        for i, (train_data, eval_data) in enumerate(eval_bar):
            self.model.fit(train_data)
            demand_0, demand_1 = self.model.prices2demand(eval_data)
            ce0 = utils.cross_entropy(demand_0.reshape(-1, 1), eval_data['Y0'].to_numpy())
            ce1 = utils.cross_entropy(demand_1.reshape(-1, 1), eval_data['Y1'].to_numpy())
            eval_bar.write(utils.getTimeStr() + ''.join(f"{key:>10.3f}" for key in [i, ce0, ce1]))
            log.append({f"Fold {i}": {'Cross Entropy For item0': ce0, 'Cross Entropy For item1': ce1}})
        return log

    