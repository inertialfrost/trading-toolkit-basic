import numpy as np
import datetime as dt
import yfinance as yf
from matplotlib import pyplot as plt
from completed import frechet as fd


def plot_chart(p: list):
    x = []
    y = []
    for i in range(0, len(p)):
        x.append(p[i][0])
        y.append(p[i][1])

    print(x)
    print(y)

    plt.plot(x, y)
    plt.show()


def simple_moving_average(p: list, t: int) -> list:
    res = []
    for i in range(0, len(p)):
        s = 0
        for j in range(max(i - t, 0), i + 1):
            s = s + p[j]
        res.append(s / min((i + 1), t + 1))
    return res


def calc_frechet_distance(scrip: str) -> float:
    stock = scrip

    sma_length = 21

    # downloading 2 months of data for 1 month analysis
    start = dt.datetime(2021, 6, 1)
    end = dt.datetime(2021, 8, 6)

    data = yf.download(stock, start=start, end=end)
    print(type(data))
    close_p = data['Close']

    close_prices = []
    # for i in range(len(close_p) - cycle_length, len(close_p)):
    #     # close_prices.append([i, close_p[i]])
    #     close_prices.append(close_p[i])

    for i in range(0, len(close_p)):
        close_prices.append(close_p[i])

    # calculating sma chart for normalisation
    simple_ma = simple_moving_average(close_prices, sma_length)

    delta = []
    for i in range(0, len(close_prices)):
        delta.append(close_prices[i] - simple_ma[i])

    pattern = fd.generate_pattern(len(close_prices), 'full sin', (max(delta) - min(delta))/2, int(len(close_prices)/sma_length))

    # single scrip analysis

    # create numpy arrays
    p = []
    for i in range(0, len(close_prices)):
        p.append([i, close_prices[i]])

    q = []
    for i in range(0, len(pattern)):
        q.append([i, pattern[i]])

    p, q = np.array(p), np.array(q)

    print('Frechet distance: ', fd.linear_frechet(p, q))

    # plot charts
    plt.plot(delta)
    plt.plot(pattern)
    plt.show()

    return fd.linear_frechet(p, q)


calc_frechet_distance('TCS.NS')