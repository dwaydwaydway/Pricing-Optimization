import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

class Preprocessor:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"run function not implemented for {self.__class__.__name__}")

class KNN_Preprocessor(Preprocessor):
    def __init__(self, param):
        super().__init__(param)

    def run(self, data):
        # Get the unknown user vector for cold starting using KNN 
        X, Y = data['train_covariate'][:len(data['train_noisy_embedding'])], data['train_noisy_embedding']
        neigh = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        neigh.fit(X, Y)
        prediction = neigh.predict(data['train_covariate'][len(data['train_noisy_embedding']):])
        data['train_noisy_embedding'] = data['train_noisy_embedding'].append(pd.DataFrame(prediction, columns=data['train_noisy_embedding'].columns), ignore_index=True)
        
        # Get the training data
        prices_decisions = data['train_prices_decisions'].drop(columns=['user_index', 'item_bought'])
        data['train_noisy_embedding'].columns = [f'user_vector_{i}' for i in range(10)]
        X = pd.concat([data['train_noisy_embedding'].reset_index(drop=True), \
                        data['train_covariate'].reset_index(drop=True), \
                        prices_decisions.reset_index(drop=True)], axis=1)
        
        # Get the training label
        Y0, Y1 = [], []
        for row in data['train_prices_decisions']['item_bought']:
            if row == -1:
                y0, y1 = 0, 0
            elif row == 0:
                y0, y1 = 1, 0
            elif row == 1:
                y0, y1 = 0, 1
            Y0.append(y0)
            Y1.append(y1)

        return {'X': X, 'Y0': pd.DataFrame(Y0), 'Y1': pd.DataFrame(Y1)}
