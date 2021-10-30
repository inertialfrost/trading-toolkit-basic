import numpy as np
# import math
import pandas as pd
from scipy.signal import argrelextrema
import datetime as dt
import yfinance as yf
from matplotlib import pyplot as plt
from stockstats import StockDataFrame

nifty50 = ['INDUSINDBK.NS', 'ADANIPORTS.NS', 'IOC.NS', 'TECHM.NS', 'TATACONSUM.NS', 'BHARTIARTL.NS', 'SBILIFE.NS', 'NTPC.NS', 'MARUTI.NS', 'TCS.NS', 'BAJAJ-AUTO.NS', 'COALINDIA.NS', 'HINDUNILVR.NS', 'HEROMOTOCO.NS', 'BPCL.NS', 'HDFCLIFE.NS', 'HDFCBANK.NS', 'KOTAKBANK.NS', 'EICHERMOT.NS', 'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'POWERGRID.NS', 'DIVISLAB.NS', 'UPL.NS', 'INFY.NS', 'SUNPHARMA.NS', 'BRITANNIA.NS', 'ONGC.NS', 'WIPRO.NS', 'DRREDDY.NS', 'ITC.NS', 'M&M.NS', 'ICICIBANK.NS', 'BAJFINANCE.NS', 'HINDALCO.NS', 'GRASIM.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 'AXISBANK.NS', 'JSWSTEEL.NS', 'SBIN.NS', 'HCLTECH.NS', 'LT.NS', 'HDFC.NS', 'TATASTEEL.NS', 'ULTRACEMCO.NS', 'SHREECEM.NS', 'RELIANCE.NS', 'CIPLA.NS']
nifty100 = ['ACC.NS', 'ABBOTINDIA.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANITRANS.NS', 'ALKEM.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AUROPHARMA.NS', 'DMART.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BANDHANBNK.NS', 'BERGEPAINT.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIOCON.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS', 'CADILAHC.NS', 'CIPLA.NS', 'COALINDIA.NS', 'COLPAL.NS', 'DLF.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GLAND.NS', 'GODREJCP.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ITC.NS', 'IOC.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INDIGO.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LTI.NS', 'LT.NS', 'LUPIN.NS', 'MRF.NS', 'M&M.NS', 'MARICO.NS', 'MARUTI.NS', 'MUTHOOTFIN.NS', 'NMDC.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'PETRONET.NS', 'PIDILITIND.NS', 'PEL.NS', 'POWERGRID.NS', 'PGHH.NS', 'PNB.NS', 'RELIANCE.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'UBL.NS', 'MCDOWELL-N.NS', 'VEDL.NS', 'WIPRO.NS', 'YESBANK.NS']

start = dt.datetime(2020, 1, 1)
# end = dt.datetime(2021, 8, 6)
end = dt.datetime.now()


def test_returns(stock: str):
    data = yf.download(stock, start=start, end=end)
    # stocks = StockDataFrame.retype(data[['Open', 'Close', 'High', 'Low', 'Volume']])

    d_returns = data['Close'].pct_change(1)
    # returns in bps
    d_returns = d_returns * 100 * 100

    # print(d_returns)

    print('Mean: ', d_returns.mean())
    print('Std Dev: ', d_returns.std())

    n_bins = 50
    plt.hist(d_returns, bins=n_bins)
    plt.show()


test_returns('TCS.NS')