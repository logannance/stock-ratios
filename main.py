# Sortino ratios of each stock in the S&P 500 as of
# 2026-03-30 covering 3 years of daily adjusted close data.
import yfinance as yf
import pandas as pd
import math
import numpy as np
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Sortino 500')
parser.add_argument('--rfr',
    type=float, help="Annual risk free rate, e.g., 0.03", required=True)

parser.add_argument('--download', action='store_true',
    help='Try to download new data as of today')

args = parser.parse_args()

if (args.download or not pathlib.Path('data.pkl').exists()):
    with open('tickers.txt', 'r') as tickers:
        symbols = [line.strip().replace('.', '-') \
            for line in tickers]

    prices = yf.download(symbols, period='3y')
    prices.to_pickle('data.pkl')
    prices.to_excel('data.xlsx')

prices = pd.read_pickle('data.pkl')
# We only care about the close price
prices = prices.drop(columns=['Open', 'Volume', 'Low', 'High'])
pct_change = prices['Close'].pct_change()

def simple_annualized_sortino(series):
    n = series.count()
    mean = series.mean() - args.rfr / 252
    down_devs = series.clip(upper=0) - mean
    down_stdev = math.sqrt((down_devs ** 2).sum() / (n - 1))

    return mean * math.sqrt(252) / down_stdev

def log_annualized_sortino(series):
    log_returns = np.log(1 + series)
    n = log_returns.count()
    mean = log_returns.mean() - math.log(1 + args.rfr) / 252
    # print(f'name: {series.name} mean: {str(mean)} n: {n}')

    down_devs = series.clip(upper=0) - mean
    down_stdev = math.sqrt((down_devs ** 2).sum() / (n - 1))
    return mean * math.sqrt(252) / down_stdev
    
simple_annualized = pct_change.agg(simple_annualized_sortino)
log_annualized = pct_change.agg(log_annualized_sortino)
df = pd.concat([simple_annualized, log_annualized], axis=1)
df.columns = ['Simple', 'Log']

df = df.sort_values('Log', ascending=False)
df.to_excel('sortinos.xlsx')