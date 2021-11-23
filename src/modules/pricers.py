import numpy as np
from tqdm import tqdm

class Pricer:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def get_revenue(self, model, x):
        proba = model.predict_proba({'X': np.array([x])})
        revenues = {}
        revenues['soft'] = (proba[0][1] * x[-2] + proba[0][2] * x[-1])
        revenues['penalized'] = revenues['soft']  * (self.alpha / (proba[0][0]+1e-08))
        revenues['hard']  = 0
        if np.argmax(proba) == 1:
            revenues['hard'] += x[-2]
        if np.argmax(proba) == 2:
            revenues['hard'] += x[-1]
        return revenues

    def run(self, model, data):
        raise NotImplementedError(f"run function not implemented for {self.__class__.__name__}")

class RandomSearchPricer(Pricer):
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

        expected_hard_revenue, expected_soft_revenue, expected_penalized_revenue, predicted_prices = [], [], [], []
        for i, x in enumerate(progress_bar):
            max_hard_revenue, max_soft_revenue, max_penalized_revenue = float('-inf'), float('-inf'), float('-inf')
            best_prices = [self.price0.min, self.price1.min]
            for price0, price1 in zip(price0_candidates, price1_candidates):
                x[-2], x[-1] = price0, price1
                revenues = self.get_revenue(model, x)
                if (self.metric == 'hard' and revenues['hard'] > max_hard_revenue)\
                    or (self.metric == 'soft' and revenues['soft'] > max_soft_revenue) \
                    or (self.metric == 'penalized' and revenues['penalized'] > max_penalized_revenue):
                    best_prices = (price0, price1)
                    max_hard_revenue = revenues['hard']
                    max_soft_revenue = revenues['soft']
                    max_penalized_revenue = revenues['penalized']
            
            predicted_prices.append(best_prices)
            expected_hard_revenue.append(max(max_hard_revenue, 0))
            expected_soft_revenue.append(max(max_soft_revenue, 0))
            expected_penalized_revenue.append(max(max_penalized_revenue, 0))
            postfix = [f'[AVG Hard Revenue: {sum(expected_hard_revenue)/(i+1):2.2f}]', \
                      f'[AVG Soft Revenue: {sum(expected_soft_revenue)/(i+1):2.2f}]', \
                      f'[AVG Penalized Revenue: {sum(expected_penalized_revenue)/(i+1):2.2f}]']
            progress_bar.set_postfix_str(''.join(postfix))

        return predicted_prices, expected_hard_revenue, expected_soft_revenue, expected_penalized_revenue

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

        expected_hard_revenue, expected_soft_revenue, expected_penalized_revenue, predicted_prices = [], [], [], []
        for i, x in enumerate(progress_bar):
            best_prices, max_hard_revenue, max_soft_revenue, max_penalized_revenue = (0, 0), 0, 0, 0

            curr_prices = [init_price0, init_price1]       
            step = self.step
            for epoch in range(self.max_epoch):
                directions = [
                    (curr_prices[0] - step, curr_prices[1]), 
                    (curr_prices[0] + step, curr_prices[1]), 
                    (curr_prices[0], curr_prices[1] - step), 
                    (curr_prices[0], curr_prices[1] + step), 
                ]
                next_hard_revenue, next_soft_revenue, next_penalized_revenue = float('-inf'), float('-inf'), float('-inf')
                next_prices = [0, 0]
                for direction in directions:
                    x[-2], x[-1] = direction[0], direction[1]
                    revenues = self.get_revenue(model, x)
                    if (self.metric == 'hard' and revenues['hard'] > next_hard_revenue)\
                        or (self.metric == 'soft' and revenues['soft'] > next_soft_revenue) \
                        or (self.metric == 'penalized' and revenues['penalized'] > next_penalized_revenue):
                        next_prices = direction
                        next_hard_revenue = revenues['hard']
                        next_soft_revenue = revenues['soft']
                        next_penalized_revenue = revenues['penalized']

                if (self.metric == 'hard' and revenues['hard'] > max_hard_revenue)\
                    or (self.metric == 'soft' and revenues['soft'] > max_soft_revenue) \
                    or (self.metric == 'penalized' and revenues['penalized'] > max_penalized_revenue):
                    best_prices = next_prices
                    max_hard_revenue = revenues['hard']
                    max_soft_revenue = revenues['soft']
                    max_penalized_revenue = revenues['penalized']

                curr_prices = next_prices
                step *= self.lambda_

            predicted_prices.append(best_prices)
            expected_hard_revenue.append(max(max_hard_revenue, 0))
            expected_soft_revenue.append(max(max_soft_revenue, 0))
            expected_penalized_revenue.append(max(max_penalized_revenue, 0))
            postfix = [f'[AVG Hard Revenue: {sum(expected_hard_revenue)/(i+1):2.2f}]', \
                      f'[AVG Soft Revenue: {sum(expected_soft_revenue)/(i+1):2.2f}]', \
                      f'[AVG Penalized Revenue: {sum(expected_penalized_revenue)/(i+1):2.2f}]']
            progress_bar.set_postfix_str(''.join(postfix))

        return predicted_prices, expected_hard_revenue, expected_soft_revenue, expected_penalized_revenue

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

        expected_soft_revenue, expected_hard_revenue, expected_penalized_revenue, predicted_prices = [], [], [], []
        for i, x in enumerate(progress_bar):
            best_prices, max_hard_revenue, max_soft_revenue, max_penalized_revenue = (0, 0), 0, 0, 0

            curr_prices = np.array([init_price0, init_price1])
            x[-2], x[-1] = init_price0, init_price1

            lr, epsilon = self.lr, self.epsilon
            beta1, beta2 = self.beta1, self.beta2

            m, v = np.array([0, 0]), np.array([0, 0])
            for epoch in range(self.max_epoch):                
                # Central Difference Gradient Estimater
                x[-2], x[-1] = (curr_prices[0] + epsilon, curr_prices[1])
                f0_x_plus_epsilon = self.get_revenue(model, x)
                x[-2], x[-1] = (curr_prices[0] - epsilon, curr_prices[1])
                f0_x_minus_epsilon = self.get_revenue(model, x)
                gradient0 = (f0_x_plus_epsilon[self.metric] - f0_x_minus_epsilon[self.metric]) / (epsilon*2)
                # Central Difference Gradient Estimater
                x[-2], x[-1] = (curr_prices[0], curr_prices[1] + epsilon)
                f1_x_plus_epsilon = self.get_revenue(model, x)
                x[-2], x[-1] = (curr_prices[0], curr_prices[1] - epsilon)
                f1_x_minus_epsilon = self.get_revenue(model, x)
                gradient1 = (f1_x_plus_epsilon[self.metric] - f1_x_minus_epsilon[self.metric]) / (epsilon*2)

                # Adam Optimizer
                gradient = np.array([gradient0, gradient1])
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * np.power(gradient, 2)
                m_hat = m / (1 - np.power(beta1, epoch+1))
                v_hat = v / (1 - np.power(beta2, epoch+1))
                curr_prices +=  lr * m_hat / (np.sqrt(v_hat) + 1e-08)

                x[-2], x[-1] = curr_prices[0], curr_prices[1]
                revenues = self.get_revenue(model, x)
                if (self.metric == 'hard' and revenues['hard'] > max_hard_revenue)\
                    or (self.metric == 'soft' and revenues['soft'] > max_soft_revenue) \
                    or (self.metric == 'penalized' and revenues['penalized'] > max_penalized_revenue):
                    best_prices = curr_prices
                    max_hard_revenue = revenues['hard']
                    max_soft_revenue = revenues['soft']
                    max_penalized_revenue = revenues['penalized']

            predicted_prices.append(best_prices)
            expected_hard_revenue.append(max(max_hard_revenue, 0))
            expected_soft_revenue.append(max(max_soft_revenue, 0))
            expected_penalized_revenue.append(max(max_penalized_revenue, 0))
            postfix = [f'[AVG Hard Revenue: {sum(expected_hard_revenue)/(i+1):2.2f}]', \
                      f'[AVG Soft Revenue: {sum(expected_soft_revenue)/(i+1):2.2f}]', \
                      f'[AVG Penalized Revenue: {sum(expected_penalized_revenue)/(i+1):2.2f}]']
            progress_bar.set_postfix_str(''.join(postfix))

        return predicted_prices, expected_hard_revenue, expected_soft_revenue, expected_penalized_revenue


