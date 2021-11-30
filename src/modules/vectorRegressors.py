import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class Regressor:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"run function not implemented for {self.__class__.__name__}")

class KNN_Regressor(Regressor):
    def __init__(self, param):
        super().__init__(param)

    def run(self, data):
        # Get the unknown user vector for cold starting using KNN 
        index_provided = data['noisy_embedding'].index
        index_pred = np.array([i for i in range(1, len(data['covariate'])+1) if i not in index_provided])
        X, Y = data['covariate'].iloc[index_provided-1], data['noisy_embedding']
        self.neigh = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
        self.neigh.fit(X, Y)
        prediction = self.neigh.predict(data['covariate'].iloc[index_pred-1])
        data['noisy_embedding'] = \
            data['noisy_embedding'].append(pd.DataFrame(prediction, 
                                                        columns=data['noisy_embedding'].columns, 
                                                        index=index_pred))
        data['noisy_embedding'] = data['noisy_embedding'].sort_index()
