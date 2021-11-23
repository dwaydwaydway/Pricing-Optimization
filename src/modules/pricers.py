import numpy as np
from tqdm import tqdm

class Pricer:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def get_revenue(self, model, x):
        proba = model.predict_proba({'X': np.array([x])})
        revenue = proba[0][1] * x[-2] + proba[0][2] * x[-1]
        return revenue

    def run(self, model, data):
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
            data['X'],
            total=len(data['X']),
            desc='[RAND_PRICEING]',
            leave=False,
            position=0)

        expected_revenue, predicted_prices = [], []
        for i, x in enumerate(progress_bar):
            max_pred_revenue = float('-inf')
            best_pred_prices = [self.price0.min, self.price1.min]
            for price0, price1 in zip(price0_candidates, price1_candidates):
                x[-2], x[-1] = price0, price1
                revenue = self.get_revenue(model, x)
                if revenue > max_pred_revenue:
                    best_pred_prices, max_pred_revenue = (price0, price1), revenue

            predicted_prices.append(best_pred_prices)
            expected_revenue.append(max(max_pred_revenue, 0))
            progress_bar.set_postfix_str(f'[AVG Revenue: {sum(expected_revenue)/(i+1):2.3f}]')

        return expected_revenue, predicted_prices

class BestStepPricer(Pricer):
    def __init__(self, param):
        super().__init__(param)

    def run(self, model, data):        
        init_price0 = (self.price0.min + self.price0.max) / 2
        init_price1 = (self.price1.min + self.price1.max) / 2
        
        progress_bar = tqdm(
            data['X'],
            total=len(data['X']),
            desc='[ZOO_PRICEING]',
            leave=False,
            position=0)

        expected_revenue, predicted_prices = [], []
        for i, x in enumerate(progress_bar):
            best_pred_prices, max_pred_revenue = 0, 0
            curr_prices = [init_price0, init_price1]
            x[-2], x[-1] = init_price0, init_price1
            curr_revenue = self.get_revenue(model, x)            
            step = self.step
            for epoch in range(self.max_epoch):
                directions = [
                    (curr_prices[0] - step,curr_prices[1]), 
                    (curr_prices[0] + step,curr_prices[1]), 
                    (curr_prices[0],curr_prices[1] - step), 
                    (curr_prices[0],curr_prices[1] + step), 
                ]
                max_revenue = float('-inf')
                next_prices = [0, 0]
                for direction in directions:
                    x[-2], x[-1] = direction[0], direction[1]
                    revenue = self.get_revenue(model, x)
                    if revenue > max_revenue:
                        next_prices = direction
                        max_revenue = revenue

                if max_revenue > max_pred_revenue:
                    best_pred_prices, max_pred_revenue = next_prices, max_revenue
                curr_revenue = max_revenue
                curr_prices = next_prices
                step *= self.lambda_

            predicted_prices.append(best_pred_prices)
            expected_revenue.append(max(max_pred_revenue, 0))
            progress_bar.set_postfix_str(f'[AVG Revenue: {sum(expected_revenue)/(i+1):2.3f}]')

        return expected_revenue, predicted_prices

class ZOOPricer(Pricer):
    def __init__(self, param):
        super().__init__(param)

    def run(self, model, data):        
        init_price0 = self.price0.min
        init_price1 = self.price1.min
        
        progress_bar = tqdm(
            data['X'],
            total=len(data['X']),
            desc='[ZOO_PRICEING]',
            leave=False,
            position=0)

        expected_revenue, predicted_prices = [], []
        for i, x in enumerate(progress_bar):
            best_pred_prices, max_pred_revenue = 0, 0

            curr_prices = np.array([init_price0, init_price1])
            x[-2], x[-1] = init_price0, init_price1
            curr_revenue = self.get_revenue(model, x)

            lr, epsilon = self.lr, self.epsilon
            beta1, beta2 = self.beta1, self.beta2
            
            m, v = np.array([0, 0]), np.array([0, 0])
            for epoch in range(self.max_epoch):

                x[-2], x[-1] = (curr_prices[0] + epsilon, curr_prices[1])
                f0_x_plus_epsilon = self.get_revenue(model, x)
                x[-2], x[-1] = (curr_prices[0] - epsilon, curr_prices[1])
                f0_x_minus_epsilon = self.get_revenue(model, x)
                gradient0 = (f0_x_plus_epsilon - f0_x_minus_epsilon) / (epsilon*2)

                x[-2], x[-1] = (curr_prices[0], curr_prices[1] + epsilon)
                f1_x_plus_epsilon = self.get_revenue(model, x)
                x[-2], x[-1] = (curr_prices[0], curr_prices[1] - epsilon)
                f1_x_minus_epsilon = self.get_revenue(model, x)
                gradient1 = (f1_x_plus_epsilon - f1_x_minus_epsilon) / (epsilon*2)

                # Adam Optimizer
                gradient = np.array([gradient0, gradient1])
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * np.power(gradient, 2)
                m_hat = m / (1 - np.power(beta1, epoch+1))
                v_hat = v / (1 - np.power(beta2, epoch+1))
                curr_prices +=  lr * m_hat / (np.sqrt(v_hat) + 1e-08)

                x[-2], x[-1] = curr_prices[0], curr_prices[1]
                curr_revenue = self.get_revenue(model, x)
                if curr_revenue > max_pred_revenue:
                    best_pred_prices, max_pred_revenue = curr_prices, curr_revenue

            predicted_prices.append(best_pred_prices)
            expected_revenue.append(max(max_pred_revenue, 0))
            progress_bar.set_postfix_str(f'[AVG Revenue: {sum(expected_revenue)/(i+1):2.3f}]')

        return expected_revenue, predicted_prices


