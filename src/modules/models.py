import numpy as np
from sklearn.linear_model import LogisticRegression
from modules.custom_model import OneVsRestLightGBMWithCustomizedLoss, FocalLoss
from sklearn.model_selection import train_test_split

class Models:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(f"fit function not implemented for {self.__class__.__name__}")

    def predict_prob(self, *args, **kwargs):
        raise NotImplementedError(f"predict_prob function not implemented for {self.__class__.__name__}")

class Logistic_Regression(Models):
    def __init__(self, param):
        super().__init__(param)

    def fit(self, data):
        self.model_0 = LogisticRegression(max_iter=self.max_iter).fit(data['X'], data['Y0'].values.ravel())
        self.model_1 = LogisticRegression(max_iter=self.max_iter).fit(data['X'], data['Y1'].values.ravel())
    
    def prices2demand(self, data):
        return self.model_0.predict_proba(data['X'])[:, 1], self.model_1.predict_proba(data['X'])[:, 1]

class lightGBM_FocalLoss(Models):
    def __init__(self, param):
        super().__init__(param)
        self.loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)

    def fit(self, data, fit_params):
        X_train, X_valid, Y_train, Y_valid = train_test_split(data['X'], data['Y'], \
                                                                test_size=0.15)
        self.model = OneVsRestLightGBMWithCustomizedLoss(loss=self.loss, n_jobs=self.n_process)
        fit_params['eval_set'] = [(X_valid, Y_valid)]
        self.model.fit(X_train, Y_train, **fit_params)
    
    def predict_proba(self, data):
        return self.model.predict_proba(data['X'])