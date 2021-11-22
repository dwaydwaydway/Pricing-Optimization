import pandas as pd
import numpy as np
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

    def run(self, data, mode='train'):
        # Get the unknown user vector for cold starting using KNN 
        index_provided = data['noisy_embedding'].index
        index_pred = np.array([i for i in range(1, len(data['covariate'])+1) if i not in index_provided])
        X, Y = data['covariate'].iloc[index_provided-1], data['noisy_embedding']
        if mode == 'train':
            self.neigh = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            self.neigh.fit(X, Y)
        prediction = self.neigh.predict(data['covariate'].iloc[index_pred-1])
        data['noisy_embedding'] = data['noisy_embedding'].append(pd.DataFrame(prediction, columns=data['noisy_embedding'].columns, index=index_pred))
        data['noisy_embedding'] = data['noisy_embedding'].sort_index()
        
        # Get the training data
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

        # Get the training label if training mode
        if mode == 'train':
            prices = data['prices_decisions'].drop(columns=['user_index', 'item_bought'])
            X = pd.concat([X, prices.reset_index(drop=True)], axis=1)
            Y = []
            for row in data['prices_decisions']['item_bought']:
                Y.append(row+1)
            return {'X': X.to_numpy(), 'Y':np.array(Y).reshape(-1)}
        elif mode == 'test':
            price_item_0, price_item_1 = pd.DataFrame({'price_item_0': [-1] * n_data}), pd.DataFrame({'price_item_1': [-1] * n_data})
            X = pd.concat([X, price_item_0, price_item_1], axis=1)
            return {'X': X.to_numpy()}
