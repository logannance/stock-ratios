# Sortino ratios of each stock in the S&P 500 as of
# 2026-03-30 covering 3 years of daily adjusted close data.
import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import pathlib
import sortino
import sharpe
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Sortino 500')
parser.add_argument('--rfr',
    type=float, help="Annual risk free rate, e.g., 0.03", required=True)

parser.add_argument('--download', action='store_true',
    help='Try to download new data as of today')

args = parser.parse_args()
sortino.rfr = args.rfr

if (args.download or not pathlib.Path('data.pkl').exists()):
    with open('tickers.txt', 'r') as tickers:
        symbols = [line.strip().replace('.', '-') \
            for line in tickers]

    data = yf.download(symbols, period='3y')
    data.to_pickle('data.pkl')
    data.to_excel('data.xlsx')

data = pd.read_pickle('data.pkl')
# We only care about the close price
data = data.drop(columns=['Open', 'Volume', 'Low', 'High'])
data = data['Close'].pct_change()

log_sharpe = data.agg(sharpe.log_annualized)

top_ten_log_sharpe = log_sharpe.nlargest(20)
top_ten_log_sharpe = top_ten_log_sharpe.sort_values(ascending=False)
top_ten_log_sharpe.plot.bar()
plt.show()

# Combine into dataframe with MultiIndex columns
df = pd.DataFrame({
    ('Sharpe', 'Simple'): data.agg(sharpe.simple_annualized),
    ('Sharpe', 'Log'): log_sharpe,
    ('Sortino', 'Simple',): data.agg(sortino.simple_annualized),
    ('Sortino', 'Log'): data.agg(sortino.log_annualized)
})

df.columns = pd.MultiIndex.from_tuples(df.columns)
df = df.sort_values(('Sharpe', 'Log'), ascending=False)
df.to_excel('ratios.xlsx')