import numpy as np
import math
import pandas as pd
import datetime as dt
import yfinance as yf
from matplotlib import pyplot as plt

import frechet as fd

scrips = ['TCS.NS']
# testing sin wave from 0 to 1 in 10 intervals
p = []
for i in range(0, 10):
    p.append([(i/10)*math.pi, 5*math.sin((i/10)*math.pi)])

q = []
for i in range(0, 10):
    q.append([(i/10)*math.pi, math.sin((i/10)*math.pi)])

# print(q)

p, q = np.array(p), np.array(q)

print('Frechet distance: ', fd.linear_frechet(p, q))