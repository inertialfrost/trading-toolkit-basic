import math
import pandas
import datetime as dt
import yfinance as yf

from scrips import nifty50, nifty100


def period_max(period: int, mid_date: dt.datetime, arr: pandas.DataFrame) -> float:
    """
    This function works on # of trading days
    :param period: integer on width of period. if odd, takes (p-1)/2 periods on both ends, else (p-1)/2-1 & (p-1)/2
    :param mid_date: date around which the data is to be scanned
    :param arr: pandas dataframe containing the exact numbers
    :return: maximum of the period
    """
    # Slicing operator works on an n - 1 basis
    # print(mid_date.date())

    if arr.index.__contains__(mid_date.date().strftime('%Y-%m-%d')):
        location = arr.index.get_loc(mid_date.date().strftime('%Y-%m-%d'))
    elif arr.index.__contains__(rel_date(mid_date, -1).date().strftime('%Y-%m-%d')):
        # print('Skip 1 day')
        location = arr.index.get_loc(rel_date(mid_date, -1).date().strftime('%Y-%m-%d'))
        # print('Skipping 1 day')
    elif arr.index.__contains__(rel_date(mid_date, -2).date().strftime('%Y-%m-%d')):
        location = arr.index.get_loc(rel_date(mid_date, -2).date().strftime('%Y-%m-%d'))
    else:
        location = arr.index.get_loc(rel_date(mid_date, -3).date().strftime('%Y-%m-%d'))
        # print('Skipping 2 days')

    if period % 2 == 0:
        ans = arr.iloc[int(location - (period - 1)/2): int(location + (period - 1)/2 + 1)].max()
        # print(arr.iloc[int(location - (period - 1)/2): int(location + (period - 1)/2 + 1)].index)
    else:
        ans = arr.iloc[int(location - (period-1)/2): int(location + (period - 1)/2 + 1)].max()
        # print(arr.iloc[int(location - (period-1)/2): int(location + (period - 1)/2 + 1)].index)
    return ans


def period_min(period: int, mid_date: dt.datetime, arr: pandas.DataFrame) -> float:
    """
    This function works on # of trading days
    :param period: integer on width of period. if odd, takes (p-1)/2 periods on both ends, else (p-1)/2-1 & (p-1)/2
    :param mid_date: date around which the data is to be scanned
    :param arr: pandas dataframe containing the exact numbers
    :return: minimum of the period
    """
    # Slicing operator works on an n - 1 basis

    if arr.index.__contains__(mid_date.date().strftime('%Y-%m-%d')):
        location = arr.index.get_loc(mid_date.date().strftime('%Y-%m-%d'))
    elif arr.index.__contains__(rel_date(mid_date, -1).date().strftime('%Y-%m-%d')):
        # print('Skip 1 day')
        location = arr.index.get_loc(rel_date(mid_date, -1).date().strftime('%Y-%m-%d'))
        # print('Skipping 1 day')
    elif arr.index.__contains__(rel_date(mid_date, -2).date().strftime('%Y-%m-%d')):
        # print('Skip 2 day')
        location = arr.index.get_loc(rel_date(mid_date, -2).date().strftime('%Y-%m-%d'))
        # print('Skipping 2 day')
    else:
        # print('Skip 3 day')
        location = arr.index.get_loc(rel_date(mid_date, -3).date().strftime('%Y-%m-%d'))
        # print('Skipping 2 days')

    if period % 2 == 0:
        ans = arr.iloc[int(location - (period - 1)/2): int(location + (period - 1)/2 + 1)].min()
    else:
        ans = arr.iloc[int(location - (period-1)/2): int(location + (period - 1)/2 + 1)].min()
    return ans


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


def main():
    scrip = 'TCS.NS'
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2021, 8, 1)
    # end = dt.datetime.now()

    scan_results = []
    i = 1
    scan_width = 25

    for scrip in ['TCS.NS']:
        scrip_data = yf.download(scrip, start=start, end=end)
        print('Length: ', len(scrip_data))
        # close = scrip_data['Close']
        low = scrip_data['Low']
        high = scrip_data['High']

        # print('(', i, '/', len(scan_list), ' ', scrip)

        # close_1_h = period_max(scan_width, t_minus(-1), high)
        # close_11_h = period_max(scan_width, t_minus(-11), high)
        # close_21_h = period_max(scan_width, t_minus(-21), high)
        # close_1_l = period_min(scan_width, t_minus(-1), low)
        # close_11_l = period_min(scan_width, t_minus(-11), low)
        # close_21_l = period_min(scan_width, t_minus(-21), low)

        for delta in range(0, 100):
            cur_date = end - dt.timedelta(days=delta)
            # print('cur_date: ', cur_date.date())
            close_l1_h = period_max(scan_width, rel_date(cur_date, -1), high)
            close_l2_h = period_max(scan_width, rel_date(cur_date, -31), high)
            close_l3_h = period_max(scan_width, rel_date(cur_date, -61), high)
            close_l1_l = period_min(scan_width, rel_date(cur_date, -1), low)
            close_l2_l = period_min(scan_width, rel_date(cur_date, -31), low)
            close_l3_l = period_min(scan_width, rel_date(cur_date, -61), low)

            asc_triangle = False
            # asc_triangle = abs((close_1_h - close_11_h)/close_1_h) <= 0.01 and \
            #     abs((close_11_h - close_21_h)/close_11_h) <= 0.01 and \
            #     close_1_l > close_11_l > close_21_l

            asc_triangle = abs((close_l1_h - close_l2_h) / close_l1_h) <= 0.01 and \
                           abs((close_l2_h - close_l3_h) / close_l2_h) <= 0.01 and \
                           close_l1_l > close_l2_l > close_l3_l

            if asc_triangle:
                scan_results.append(cur_date.date())
                print('Pass')
                # print('t_minus_1:', t_minus(-1))
                # print('close_1_l: ', close_1_l)
                # print('close_1_h: ', close_1_h)
                # print('t_minus_11:', t_minus(-11))
                # print('close_11_l: ', close_11_l)
                # print('close_11_h: ', close_11_h)
                # print('t_minus_21:', t_minus(-21))
                # print('close_21_l: ', close_21_l)
                # print('close_21_h: ', close_21_h)

                print('t_minus_1:', rel_date(cur_date, -1).date().strftime('%d-%m-%Y'))
                print('close_1_l: ', close_l1_l)
                print('close_1_h: ', close_l1_h)
                print('t_minus_11:', rel_date(cur_date, -31).date().strftime('%d-%m-%Y'))
                print('close_11_l: ', close_l2_l)
                print('close_11_h: ', close_l2_h)
                print('t_minus_21:', rel_date(cur_date, -61).date().strftime('%d-%m-%Y'))
                print('close_21_l: ', close_l3_l)
                print('close_21_h: ', close_l3_h)
            else:
                # print('Fail')
                pass

            i = i + 1

    print('Results: ')
    print(scan_results)


main()