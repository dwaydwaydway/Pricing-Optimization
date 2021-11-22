import numpy as np
from tqdm import tqdm

class Pricer:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"run function not implemented for {self.__class__.__name__}")

class RandomPricer(Pricer):
    def __init__(self, param):
        super().__init__(param)

    def run(self, model, data):
        price0_candidates = np.random.uniform(self.price0.min, \
                                                self.price0.max, \
                                                self.n_search)
        price1_candidates = np.random.uniform(self.price1.min, \
                                                self.price1.max, \
                                                self.n_search)
        progress_bar = tqdm(
            zip(data['X'], data['Y']),
            total=len(data['X']),
            desc='[RAND_PRICEING]',
            leave=False,
            position=0)

        expected_revenue = 0
        predicted_prices = []
        for i, (x, y) in enumerate(progress_bar):
            max_pred_revenue = float('-inf')
            proba = model.predict_proba({'X': np.array([x])})
            prices = [self.price0.min, self.price1.min]
            for price0, price1 in zip(price0_candidates, price1_candidates):
                x[-2], x[-1] = price0, price1
                proba = model.predict_proba({'X': np.array([x])})
                if np.argmax(proba) == 1 and price0 > max_pred_revenue:
                    prices[0] = max_pred_revenue = price0
                elif np.argmax(proba) == 2 and price1 > max_pred_revenue:
                    prices[1] = max_pred_revenue = price1

            predicted_prices.append(prices)
            expected_revenue += max(max_pred_revenue, 0)
            progress_bar.set_postfix_str(f'[AVG Revenue: {expected_revenue/(i+1):2.3f}]')

        return expected_revenue / len(data["X"]), predicted_prices