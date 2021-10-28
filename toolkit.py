from typing import List, Union, Any
import pandas
import datetime as dt
import csv
import sys
from tabulate import tabulate

import pandas as pd
import yfinance as yf

from scrips import nifty50, nifty100, nifty200, nifty500


def period_max(period: int, ref_loc: int, data: pandas.DataFrame) -> (float, int):
    """
    Simple period max based on the index provided, looks back and can take end of data
    :param period: period for which search is done
    :param ref_loc: integer index of where to search, zero to len - 1
    :param data: relevant pandas dataframe
    :return: maximum of the array in the period, index location of the max
    """
    # print('Test (period_max)---------', period, ref_loc)
    max_value = data.iloc[ref_loc - period + 1: ref_loc].max()
    max_value_loc = data.index.get_loc(data.iloc[ref_loc - (period - 1): ref_loc].idxmax(axis=0))
    max_value_idx = data.iloc[ref_loc - (period - 1): ref_loc].idxmax(axis=0)
    return max_value, max_value_loc, max_value_idx


def period_min(period: int, ref_loc: int, data: pandas.DataFrame) -> (float, int):
    """
    Simple period min based on the index provided, looks back and can take end of data
    :param period: period for which search is done
    :param ref_loc: integer index of where to search, zero to len - 1
    :param data: relevant pandas dataframe
    :return: min of the array in the period, index location of the max
    """
    # print('Test (period_max)---------', period, ref_loc)
    min_value = data.iloc[ref_loc - (period - 1): ref_loc].min()
    min_value_loc = data.index.get_loc(data.iloc[ref_loc - (period - 1): ref_loc].idxmin(axis=0))
    min_value_idx = data.iloc[ref_loc - (period - 1): ref_loc].idxmin(axis=0)
    return min_value, min_value_loc, min_value_idx


def period_max_date(period: int, ref_date: dt.datetime, arr: pandas.DataFrame) -> (float, int):
    """
    This function works on # of trading days
    :param period: integer on width of period. if odd, takes (p-1)/2 periods on both ends, else (p-1)/2-1 & (p-1)/2
    :param ref_date: date around which the data is to be scanned
    :param arr: pandas dataframe containing the exact numbers
    :return: maximum of the period
    """
    # Slicing operator works on an n - 1 basis
    # print(mid_date.date())

    if arr.index.__contains__(ref_date.date().strftime('%Y-%m-%d')):
        location = arr.index.get_loc(ref_date.date().strftime('%Y-%m-%d'))
    elif arr.index.__contains__(rel_date(ref_date, -1).date().strftime('%Y-%m-%d')):
        # print('Skip 1 day')
        location = arr.index.get_loc(rel_date(ref_date, -1).date().strftime('%Y-%m-%d'))
        # print('Skipping 1 day')
    elif arr.index.__contains__(rel_date(ref_date, -2).date().strftime('%Y-%m-%d')):
        location = arr.index.get_loc(rel_date(ref_date, -2).date().strftime('%Y-%m-%d'))
    else:
        location = arr.index.get_loc(rel_date(ref_date, -3).date().strftime('%Y-%m-%d'))
        # print('Skipping 2 days')

    f_loc = 0

    ans = arr.iloc[int(location - (period - 1)): int(location)].max()
    f_loc = arr.iloc[int(location - (period - 1)): int(location)].idxmax(axis=0)

    return ans, f_loc


def period_min_date(period: int, ref_date: dt.datetime, arr: pandas.DataFrame) -> float:
    """
    This function works on # of trading days
    :param period: integer on width of period. if odd, takes (p-1)/2 periods on both ends, else (p-1)/2-1 & (p-1)/2
    :param ref_date: date around which the data is to be scanned
    :param arr: pandas dataframe containing the exact numbers
    :return: minimum of the period
    """
    # Slicing operator works on an n - 1 basis

    if arr.index.__contains__(ref_date.date().strftime('%Y-%m-%d')):
        location = arr.index.get_loc(ref_date.date().strftime('%Y-%m-%d'))
    elif arr.index.__contains__(rel_date(ref_date, -1).date().strftime('%Y-%m-%d')):
        # print('Skip 1 day')
        location = arr.index.get_loc(rel_date(ref_date, -1).date().strftime('%Y-%m-%d'))
        # print('Skipping 1 day')
    elif arr.index.__contains__(rel_date(ref_date, -2).date().strftime('%Y-%m-%d')):
        # print('Skip 2 day')
        location = arr.index.get_loc(rel_date(ref_date, -2).date().strftime('%Y-%m-%d'))
        # print('Skipping 2 day')
    else:
        # print('Skip 3 day')
        location = arr.index.get_loc(rel_date(ref_date, -3).date().strftime('%Y-%m-%d'))
        # print('Skipping 2 days')

    f_loc = 0

    ans = arr.iloc[int(location - (period - 1)): int(location)].min()
    f_loc = arr.iloc[int(location - (period - 1)): int(location)].idxmin(axis=0)

    return ans, f_loc


def rel_date(date: dt.datetime, delta: int) -> dt.datetime:
    """
    Returns a datetime that is date offset by delta
    :param date: starting date
    :param delta: offset, can be negative or positive
    :return: offset datetime
    """
    return date + dt.timedelta(days=delta)


def t_minus(delta: int) -> dt.datetime:
    """
    Same as rel_date but uses current date as end, equivalent to Close(-n), more useful for scanners
    :param delta:
    :return:
    """
    return dt.datetime.now() + dt.timedelta(days=delta)
    # return dt.datetime(2021, 8, 1) + dt.timedelta(days=delta)


# this function needs to take in list of scrips, pattern (function or expression), time to find the pattern
# pattern_backtester needs to take in all of above and success criteria, and does historical backtesting
def pattern_asc_triag(data: pandas.DataFrame, period: int, start_loc: int) -> list:
    """
    Identifies ascending triangles/wedges from start_loc backwards, over period candles
    :param data: Pandas dataframe, whole block
    :param start_loc: index location from which search has to start backwards
    :param period: period over which search has to be done
    :return:
    """
    # Algorithm
    # Assumption: We are on the rising edge after the first local minima
    # Step 1: Scan for p_min(low) in period/4. Assign that as min1, loc_min1
    # Step 2: From loc_min1, scan for p_max(high) in period/2. Assign that as max1, loc_max1
    # Step 3: From loc_max1, scan for p_min(low) in period/2. Assign that as min2, loc_min2
    # Step 4: From loc_min2, scan for p_max(high) in period/2. Assign that as max2, loc_max2
    # Step 5: If max1/max2 ranges in (0.995, 1.005) and min1 > min2 => Ascending Triangle Possible

    scrip_list = data.columns.levels[1].to_list()
    n = start_loc
    result = []

    for scrip in scrip_list:
        min1, loc_min1, idx_min1 = period_min(int(period/4), n, data['Low'][scrip])
        max1, loc_max1, idx_max1 = period_max(int(period/3), loc_min1, data['High'][scrip])
        min2, loc_min2, idx_min2 = period_min(int(period/3), loc_max1, data['Low'][scrip])
        max2, loc_max2, idx_max2 = period_max(int(period/3), loc_min2, data['High'][scrip])

        if 0.995 <= max1/max2 <= 1.005 and min1 > min2:
            result.append([scrip, min1, idx_min1.strftime('%Y-%m-%d'), max1, idx_max1.strftime('%Y-%m-%d'),
                           min2, idx_min2.strftime('%Y-%m-%d'), max2, idx_max2.strftime('%Y-%m-%d')])
    print('\n', 'Ascending Triangle Scan', '\n')
    print(tabulate(result, headers=['Scrip', 'Min1', 'Idx_Min1', 'Max1', 'Idx_Max1', 'Min2', 'Idx_Min2', 'Max2',
                                    'Idx_Max2']))
    return result


def pattern_asc_triangle_date(data: pd.DataFrame, scrip_list: list, cur_date: dt.datetime) -> list:
    """
    Period min/max based pattern scanner, returns list of name
    :param data:
    :param scrip_list:
    :param cur_date:
    :return:
    """

    scan_results = []
    final_result = []
    i = 1
    scan_width = 10
    t_0 = -1
    t_1 = -11
    t_2 = -21

    for scrip in scrip_list:
        # scrip_data = yf.download(scrip, start=start, end=end)
        # print('Length: ', len(scrip_data))
        # close = scrip_data['Close']
        low = data['Low'][scrip]
        high = data['High'][scrip]

        # print('(', i, '/', len(scan_list), ' ', scrip)

        # print('cur_date: ', cur_date.date())
        close_l1_h, loc_l1_h = period_max_date(scan_width, rel_date(cur_date, t_0), high)
        close_l2_h, loc_l2_h = period_max_date(scan_width, rel_date(cur_date, t_1), high)
        close_l3_h, loc_l3_h = period_max_date(scan_width, rel_date(cur_date, t_2), high)
        close_l1_l, loc_l1_l = period_min_date(scan_width, rel_date(cur_date, t_0), low)
        close_l2_l, loc_l2_l = period_min_date(scan_width, rel_date(cur_date, t_1), low)
        close_l3_l, loc_l3_l = period_min_date(scan_width, rel_date(cur_date, t_2), low)

        asc_triangle = False
        # asc_triangle = abs((close_1_h - close_11_h)/close_1_h) <= 0.01 and \
        #     abs((close_11_h - close_21_h)/close_11_h) <= 0.01 and \
        #     close_1_l > close_11_l > close_21_l

        asc_triangle = abs((close_l1_h - close_l2_h) / close_l1_h) <= 0.01 and \
                       abs((close_l2_h - close_l3_h) / close_l2_h) <= 0.01 and \
                       abs((close_l1_h - close_l3_h) / close_l3_h) <= 0.01 and \
                       close_l1_l > close_l2_l > close_l3_l

        if asc_triangle:
            final_result.append(scrip)

    print('\n', 'Pattern Matching Asc Triangle', '\n', '------------------------------------')
    for item in final_result:
        print(item)
    return final_result


# this function calculates the ratio of volumes of up days and down days, if > 1 accumulation possible
def detect_accumulation():
    # scrip_list = ['TCS.NS', 'ADANIPORTS.NS', 'APLAPOLLO.NS', 'POWERGRID.NS']
    scrip_list = nifty100
    start = dt.datetime(2020, 8, 31)
    end = dt.datetime(2021, 7, 13)

    scrip_data = yf.download(scrip_list, start=start, end=end)
    # print(scrip_data)
    # print(scrip_data.loc[start.date().strftime('%Y-%m-%d')]['Volume']['ADANIPORTS.NS'])
    # print(len(scrip_data.iloc[0]))
    # print(scrip_data.iloc[scrip_data.shape[0] - 1])

    result = []
    analysis_period = 100
    up_vol = 0
    down_vol = 0
    lp = scrip_data.shape[0] - 1

    for scrip in scrip_list:
        print('Initiating: ', scrip)
        up_vol = 0
        down_vol = 0
        for i in range(0, analysis_period):
            # print('lp ', lp, 'i ', i)
            opn = scrip_data.iloc[lp - i]['Open'][scrip]
            cls = scrip_data.iloc[lp - i]['Close'][scrip]

            if cls >= opn:
                up_vol = up_vol + scrip_data.iloc[lp - i]['Volume'][scrip]
            elif opn > cls:
                down_vol = down_vol + scrip_data.iloc[lp - i]['Volume'][scrip]

        if up_vol / down_vol >= 1.25:
            result.append([scrip, up_vol / down_vol])

    print(result)


"""
Candlestick patterns
- (task) hammers need confirmation post occurance, ideally should be checked for 2-3 days before, along with higher
  than normal volumes
"""


def candlestick_hammer_b(data: pd.DataFrame, scrip_list: list, shortness: float, topwick: float) -> list:
    """
    :param data:
    :param scrip_list:
    :param shortness: proportion of the length of the candle body (0, 1)
    :param topwick: proportion of the length of the top wick (0, 1)
    :return:
    """
    result = []
    l = data.shape[0] - 1
    for scrip in scrip_list:
        # candle green, then short, with top wick very small
        open = data.iloc[l]['Open'][scrip]
        close = data.iloc[l]['Close'][scrip]
        high = data.iloc[l]['High'][scrip]
        low = data.iloc[l]['Low'][scrip]
        if close > open:
            if (close - open) <= (high - low) * shortness:
                if (high - close)/(high - low) <= topwick:
                    result.append([scrip, open, high, low, close])

    print('\n', 'Bullish Hammer Scrips', '\n', '------------------------------------')
    for row in result:
        print(row)
    return result


def candlestick_inv_hammer_b(data: pd.DataFrame, scrip_list: list, shortness: float, topwick: float) -> list:
    """
    :param data:
    :param scrip_list:
    :param shortness: proportion of the length of the candle body (0, 1)
    :param topwick: proportion of the length of the top wick (0, 1)
    :return:
    """

    result = []
    l = data.shape[0] - 1
    for scrip in scrip_list:
        # candle green, then short, with top wick very small
        open = data.iloc[l]['Open'][scrip]
        close = data.iloc[l]['Close'][scrip]
        high = data.iloc[l]['High'][scrip]
        low = data.iloc[l]['Low'][scrip]
        if close > open:
            if (close - open) <= (high - low) * shortness:
                if (close - low)/(high - low) <= topwick:
                    result.append([scrip, open, high, low, close])

    print('\n', 'Bullish Inverted Hammer Scrips', '\n', '------------------------------------')
    for row in result:
        print(row)
    return result


def candlestick_two_white_soldiers(data: pd.DataFrame, scrip_list: list) -> list:
    # The pattern consists of three consecutive long-bodied candlesticks that open
    # within the previous candle's real body and a close that exceeds the previous candle's high. These candlesticks
    # should not have very long shadows and ideally open within the real body of the preceding candle in the pattern.

    result = []
    length = data.shape[0] - 1

    for scrip in scrip_list:
        i = length
        second_c_green = data.iloc[i - 1]['Open'][scrip] < data.iloc[i - 1]['Close'][scrip]
        first_c_green = data.iloc[i]['Open'][scrip] < data.iloc[i]['Close'][scrip]

        if first_c_green and second_c_green:
            first_c_long = (data.iloc[i]['Close'][scrip] - data.iloc[i]['Open'][scrip]) >= \
                           (data.iloc[i]['High'][scrip] - data.iloc[i]['Low'][scrip]) * 0.60

            second_c_long = (data.iloc[i - 1]['Close'][scrip] - data.iloc[i - 1]['Open'][scrip]) >= \
                            (data.iloc[i - 1]['High'][scrip] - data.iloc[i - 1]['Low'][scrip]) * 0.60

            if first_c_long and second_c_long:
                first_c_higher = data.iloc[i - 1]['Open'][scrip] < data.iloc[i]['Open'][scrip] \
                                 and \
                                 data.iloc[i - 1]['Close'][scrip] < data.iloc[i]['Close'][scrip]

                if first_c_higher:
                    result.append(scrip)
    print('\n', 'Two White Soldiers', '\n', '------------------------------------')
    for row in result:
        print(row)
    return result


def scan_watchlist() -> list:
    """
    Reads a file with below columns
    1. yf compatible scrip
    2. trigger price
    3. margin: fraction, within how much should the trigger activate
    4. lookout: bounce (+ve before hitting trigger), cross (+ve before hitting trigger) [clarify]
    5. comment: note on what is the thought process behind the trigger
    :return:
    return or print a table of below format
    scrip trigger margin actual_price actual_margin direction comment
    """

    watchlist_file = open('watchlist.csv')
    csvreader = csv.reader(watchlist_file)

    wl_rows = []
    wl_header = next(csvreader)

    for row in csvreader:
        wl_rows.append(row)

    wl_scrips = []
    for row in wl_rows:
        wl_scrips.append(row[0])

    end = dt.datetime.now()
    start = rel_date(end, -4)

    data = yf.download(wl_scrips, start=start, end=end)

    # trigger_hits = [['scrip', 'trigger', 'margin', 'act_price', 'act_margin', 'trigger_type', 'scope', 'comment']]
    trigger_hits = []
    for row in wl_rows:
        if data.iloc[data.shape[0] - 1]['Low'][row[0]] <= float(row[1]) <= data.iloc[data.shape[0] - 1]['High'][row[0]]:
            trigger_hits.append([row[0], row[1], row[2], "%.2f" % data.iloc[data.shape[0] - 1]['Close'][row[0]],
                                '--', 'in range', row[3], row[4]])
        elif abs(data.iloc[data.shape[0] - 1]['Low'][row[0]] - float(row[1]) / float(row[1])) <= abs(float(row[2])) or \
                abs(data.iloc[data.shape[0] - 1]['High'][row[0]] - float(row[1]) / float(row[1])) <= float(row[2]):
            trigger_hits.append([row[0], row[1], row[2], "%.2f" % data.iloc[data.shape[0] - 1]['Close'][row[0]],
                                 "%.2f" % ((float(row[1]) - data.iloc[data.shape[0] - 1]['Close'][row[0]]) / float(row[1])),
                                 'approaching', row[3], row[4]])

    print('\n', 'Watchlist results (', len(trigger_hits), '/', len(wl_scrips), ') hit', '\n')
    print(tabulate(trigger_hits, headers=['scrip', 'trigger', 'margin', 'act_price', 'act_margin', 'trigger_type',
                                          'scope', 'comment']))
    return trigger_hits


def gather_data(scrip_list: list, num_days: int) -> pd.DataFrame:
    end = dt.datetime.now()
    # end = dt.datetime(2021, 9, 1)
    start = rel_date(end, num_days * -1)

    data = yf.download(scrip_list, start=start, end=end)
    # data.to_csv('stock_data.csv')
    return data


def main():
    arguments = sys.argv
    # scrip_list = ['TCS.NS', 'INFY.NS']
    scrip_list = nifty200
    data_downloaded = False
    verbose = False

    if 'verbose' in sys.argv:
        verbose = True

    if 'help' in sys.argv:
        help_string = [
            ['scan_watchlist', 'Scans watchlist from watchlist.csv in local folder'],
            ['scan_bull_hammer', 'Scans for bullish hammers'],
            ['scan_inv_bull_hammer', 'Scans for inverted bullish hammers'],
            ['two_white_soldiers', 'Scans for two white soldiers'],
            ['pattern_asc_tri', 'Scans for ascending triangles'],
        ]
        for item in help_string:
            print(item)

    if 'scan_watchlist' in sys.argv:
        trig_hits = scan_watchlist()

    # data = gather_data(scrip_list, 100)
    data = None
    if 'scan_bull_hammer' in sys.argv:
        if not data_downloaded:
            data = gather_data(scrip_list, 100)
            data_downloaded = True
        bullish_hammers = candlestick_hammer_b(data, scrip_list, 0.4, 0.1)

    if 'scan_inv_bull_hammer' in sys.argv:
        if not data_downloaded:
            data = gather_data(scrip_list, 100)
            data_downloaded = True
        inverted_bullish_hammers = candlestick_inv_hammer_b(data, scrip_list, 0.4, 0.1)

    if 'scan_two_white_soldiers' in sys.argv:
        if not data_downloaded:
            data = gather_data(scrip_list, 100)
            data_downloaded = True
        two_white_soldiers = candlestick_two_white_soldiers(data, scrip_list)

    if 'pattern_asc_tri' in sys.argv:
        if not data_downloaded:
            data = gather_data(scrip_list, 100)
            data_downloaded = True
        asc_triag = pattern_asc_triag(data, 50, data.shape[0] - 1)


main()
