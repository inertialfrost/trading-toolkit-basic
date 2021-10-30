import numpy as np
# import math
import pandas as pd
from scipy.signal import argrelextrema
import datetime as dt
import yfinance as yf
from matplotlib import pyplot as plt
from stockstats import StockDataFrame

"""
Level 1: Get all swing trade indicators for NIFTY 50, and filter out ones which are good for swing trade
1.1: Get all technical indicators, populate in an array
1.1.1: Indicators used: SMA Cross Over, EOM, RSI
"""
nifty50 = ['INDUSINDBK.NS', 'ADANIPORTS.NS', 'IOC.NS', 'TECHM.NS', 'TATACONSUM.NS', 'BHARTIARTL.NS', 'SBILIFE.NS', 'NTPC.NS', 'MARUTI.NS', 'TCS.NS', 'BAJAJ-AUTO.NS', 'COALINDIA.NS', 'HINDUNILVR.NS', 'HEROMOTOCO.NS', 'BPCL.NS', 'HDFCLIFE.NS', 'HDFCBANK.NS', 'KOTAKBANK.NS', 'EICHERMOT.NS', 'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'POWERGRID.NS', 'DIVISLAB.NS', 'UPL.NS', 'INFY.NS', 'SUNPHARMA.NS', 'BRITANNIA.NS', 'ONGC.NS', 'WIPRO.NS', 'DRREDDY.NS', 'ITC.NS', 'M&M.NS', 'ICICIBANK.NS', 'BAJFINANCE.NS', 'HINDALCO.NS', 'GRASIM.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 'AXISBANK.NS', 'JSWSTEEL.NS', 'SBIN.NS', 'HCLTECH.NS', 'LT.NS', 'HDFC.NS', 'TATASTEEL.NS', 'ULTRACEMCO.NS', 'SHREECEM.NS', 'RELIANCE.NS', 'CIPLA.NS']
nifty100 = ['ACC.NS', 'ABBOTINDIA.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANITRANS.NS', 'ALKEM.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AUROPHARMA.NS', 'DMART.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BANDHANBNK.NS', 'BERGEPAINT.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BIOCON.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS', 'CADILAHC.NS', 'CIPLA.NS', 'COALINDIA.NS', 'COLPAL.NS', 'DLF.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GLAND.NS', 'GODREJCP.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ITC.NS', 'IOC.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INDIGO.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LTI.NS', 'LT.NS', 'LUPIN.NS', 'MRF.NS', 'M&M.NS', 'MARICO.NS', 'MARUTI.NS', 'MUTHOOTFIN.NS', 'NMDC.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'PETRONET.NS', 'PIDILITIND.NS', 'PEL.NS', 'POWERGRID.NS', 'PGHH.NS', 'PNB.NS', 'RELIANCE.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'UBL.NS', 'MCDOWELL-N.NS', 'VEDL.NS', 'WIPRO.NS', 'YESBANK.NS']


def main():
    # stock_list = ['TCS.NS', 'INFY.NS', 'TECHM.NS', 'LTI.NS', 'MPHASIS.NS', 'NAUKRI.NS', 'COFORGE.NS', 'MINDTREE.NS',
    #          'WIPRO.NS', 'INFY.NS', 'HCLTECH.NS']

    testList = ['TCS.NS']

    stock_list = testList
    indicators = {}

    start = dt.datetime(2021, 6, 1)
    # end = dt.datetime(2021, 8, 6)
    end = dt.datetime.now()

    # Loop should start here
    for stock in stock_list:
        data = yf.download(stock, start=start, end=end)
        print(type(data))
        print('Data downloaded for  ', stock)
        stocks = StockDataFrame.retype(data[['Open', 'Close', 'High', 'Low', 'Volume']])
        indicators[stock] = {
            'sma_10': [],
            'sma_20': [],
            'eom': [],
            'rsi_14': []
        }

        indicators[stock]['sma_10'] = stocks['close_10_sma']
        indicators[stock]['sma_20'] = stocks['close_20_sma']
        indicators[stock]['rsi_14'] = stocks['rsi_14']

        ilocs_min = argrelextrema(np.array(data['Low']), np.less_equal, order=3)[0]
        ilocs_max = argrelextrema(np.array(data['High']), np.greater_equal, order=3)[0]

        print('Maxima at ')
        print(ilocs_max)

        print(type(ilocs_max))

        # print(type(indicators[stock]['sma_10']))
        # plt.plot(stocks['close_16_sma'], color='b', label='SMA')

        min_data = []
        max_data = []
        max_data_dates = []
        min_data_dates = []

        for x in ilocs_max:
            max_data_dates.append(data['High'].index.values[x])
            max_data.append(data['High'][x])

        for x in ilocs_min:
            min_data_dates.append(data['Low'].index.values[x])
            min_data.append(data['Low'][x])

        plt.plot(data.Close, color='g', label='Close Prices')
        plt.scatter(max_data_dates, max_data, marker='^')
        plt.scatter(min_data_dates, min_data, marker='v')
        plt.legend(loc='lower right')
        plt.show()


        # print('Indicators populated for ', stock)

        # if len(stock_list) <= 2:
        #     data['min'] =

    # print(indicators[stock_list[0]]['sma_10'])
    # filtering stocks
    positive_stocks = []
    for stock in indicators:
        sma_flag = False
        rsi_flag = False

        # print('Indicators for ', stock)
        # print('SMA 10 vs SMA 20: %8.2f, %8.2f' % (indicators[stock]['sma_10'].iloc[-1], indicators[stock]['sma_20'].iloc[-1]))
        # print('RSI 14: %8.2f ' % indicators[stock]['rsi_14'].iloc[-1])
        # print('.....................................................')

        if indicators[stock]['sma_10'].iloc[-1] >= indicators[stock]['sma_20'].iloc[-1]:
            sma_flag = True
        if indicators[stock]['rsi_14'].iloc[-1] <= 30:
            rsi_flag = True

        # if sma_flag == True and rsi_flag == True:
        #     positive_stocks.append(stock)

        if rsi_flag is True:
            positive_stocks.append(stock)

    print('Final Stocks are...')
    # print(positive_stocks)

    for stock in positive_stocks:
        print('Indicators for ', stock)
        print('SMA 10 vs SMA 20: %8.2f, %8.2f' % (indicators[stock]['sma_10'].iloc[-1], indicators[stock]['sma_20'].iloc[-1]))
        print('RSI 14: %8.2f ' % indicators[stock]['rsi_14'].iloc[-1])
        print('.....................................................')


def test():
    testList = ['TCS.NS']

    stock_list = testList
    indicators = {}

    start = dt.datetime(2021, 1, 1)
    # end = dt.datetime(2021, 8, 6)
    end = dt.datetime.now()

    for stock in stock_list:
        data = yf.download(stock, start=start, end=end)
        print(type(data))
        print('Data downloaded for  ', stock)
        stocks = StockDataFrame.retype(data[['Open', 'Close', 'High', 'Low', 'Volume']])
        indicators[stock] = {
            'sma_10': [],
            'sma_20': [],
            'eom': [],
            'rsi_14': []
        }

        # indicators[stock]['sma_10'] = stocks['close_10_sma']
        # indicators[stock]['sma_20'] = stocks['close_20_sma']
        # indicators[stock]['rsi_14'] = stocks['rsi_14']

        ilocs_min = argrelextrema(np.array(data['Low']), np.less_equal, order=3)[0]
        ilocs_max = argrelextrema(np.array(data['High']), np.greater_equal, order=3)[0]

        print('Maxima at ')
        print(ilocs_max)

        print(type(ilocs_max))

        # print(type(indicators[stock]['sma_10']))
        # plt.plot(stocks['close_16_sma'], color='b', label='SMA')

        min_data = []
        max_data = []
        max_data_dates = []
        min_data_dates = []

        for x in ilocs_max:
            max_data_dates.append(data['High'].index.values[x])
            max_data.append(data['High'][x])

        for x in ilocs_min:
            min_data_dates.append(data['Low'].index.values[x])
            min_data.append(data['Low'][x])

        plt.plot(data.High, color='g', label='High Prices')
        plt.plot(data.Low, color='r', label='Low Prices')
        plt.scatter(max_data_dates, max_data, marker='^')
        plt.scatter(min_data_dates, min_data, marker='v')
        plt.legend(loc='lower left')
        plt.show()


# main()
test()