import numpy
import pickle
import random
import pandas as pd
import numpy as np
from time import gmtime, strftime


def openData(filePath):
    if filePath.suffix == '.csv':
        loaded = pd.read_csv(filePath)
    else:
        with open(filePath, "rb") as readfile:
            loaded = pickle.load(readfile)
    return loaded

def get_revenue_maximizing_price_and_revenue(price_options, demand_predictions):
    revenues = [i*j for i, j in zip(price_options, demand_predictions)]   
    return price_options[np.argmax(revenues)], max(revenues)

def get_KFold(data, k):
    n_data = [len(value) for key, value in data.items()][0]
    indices = list(range(n_data))
    random.shuffle(indices)
    shuffled_data = {}
    for key in data:
        data[key] = data[key].reindex(indices)
    for i in range(0, n_data, n_data//k):
        train_folds, eval_fold = {}, {}
        for key, value in data.items():
            train_folds[key] = pd.concat((value[0:i], value[i+n_data//k:]))
            eval_fold[key] = value[i:i+n_data//k]
        yield train_folds, eval_fold

def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce

def getTimeStr():
    return f"[{strftime('%Y-%m-%d %H:%M:%S', gmtime())}] "