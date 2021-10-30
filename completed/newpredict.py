import yfinance as yf
import datetime as dt

import numpy as np

company = 'TCS.NS'
index = 'NSEI'
start = dt.datetime(2021, 6, 1)
end = dt.datetime.now()

data = yf.download(company, start, end)
i_data = yf.download(index, start, end)

# Prepare data

d_returns = []
for i in range(0, len(data)):
    d_returns.append((data['High'][i] - data['Low'][i])/data['Low'][i])

d_vol = data['Volume']

d_returns = np.array(d_returns)
d_vol = np.array(d_vol)

r = np.corrcoef(d_returns, d_vol)
print('Return and Volume correlation')
print(r)

d_stock = data['Close']
d_index = i_data['Close']

r = np.corrcoef(d_stock, d_index)
print('Stock and Index correlation')
print(r)
