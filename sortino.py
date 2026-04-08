import pandas as pd
import math
import numpy as np

# args will be injected from main.py
rfr = None

def simple_annualized(series):
    n = series.count()
    daily_rfr = rfr / 252
    mean = series.mean() - daily_rfr
    down_devs = (series - daily_rfr).clip(upper=0)
    down_stdev = math.sqrt((down_devs ** 2).sum() / (n - 1))

    return mean * math.sqrt(252) / down_stdev

def log_annualized(series): 
    log_returns = np.log(1 + series)
    n = log_returns.count()
    log_daily_rfr = math.log(1 + rfr) / 252
    mean = log_returns.mean() - log_daily_rfr

    # print(f'name: {series.name} mean: {str(mean)} n: {n}')
    down_devs = (series - log_daily_rfr).clip(upper=0)
    down_stdev = math.sqrt((down_devs ** 2).sum() / (n - 1))
    return mean * math.sqrt(252) / down_stdev