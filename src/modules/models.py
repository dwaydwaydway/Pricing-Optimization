from sklearn.linear_model import LogisticRegression

class Models:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(f"fit function not implemented for {self.__class__.__name__}")

    def prices2demand(self, *args, **kwargs):
        raise NotImplementedError(f"prices2demand function not implemented for {self.__class__.__name__}")

class Logistic_Regression(Models):
    def __init__(self, param):
        super().__init__(param)

    def fit(self, data):
        self.model_0 = LogisticRegression(max_iter=self.max_iter).fit(data['X'], data['Y0'].values.ravel())
        self.model_1 = LogisticRegression(max_iter=self.max_iter).fit(data['X'], data['Y1'].values.ravel())
    
    def prices2demand(self, data):
        return self.model_0.predict_proba(data['X'])[:, 1], self.model_1.predict_proba(data['X'])[:, 1]
