import numpy as np
import pandas as pd
import math

def simple_annualized(series):
    return series.mean() / series.std() * math.sqrt(252)

def log_annualized(series):
    log_returns = np.log(1 + series)
    return log_returns.mean() / log_returns.std() * math.sqrt(252)

def log_annualized_theoretical_weight(series):
    sum = series.where(series >= 1, 0).sum()
    return series / sum