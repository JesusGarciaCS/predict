
import numpy as np
import matplotlib as mpl
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from scipy import stats
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
from statistics import mean

start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2019, 1, 9)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()
# dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],
# 'yahoo', start = start, end = end)['Adj Close']
# print(dfcomp)

# correlation
# retscomp = dfcomp.pct_change()
# corr = retscomp.corr()
# print(corr)

# scatter plot
# plt.scatter(retscomp. AAPL, retscomp. GE)
# plt.xlabel('Returns AAPL')
# plt.ylabel('Returns GE')
# plt.show()

# scatter_matrix
# style.use('ggplot')
# scatter_matrix(retscomp, diagonal='kde', alpha=0.2, figsize=(10, 10))
# plt.show()

# HeatMap-- lighter the color, the more correlated the two stocks are.
# plt.imshow(corr, cmap='hot', interpolation='none')
# plt.colorbar()
# plt.xticks(range(len(corr)), corr.columns)
# plt.yticks(range(len(corr)), corr.columns)
# plt.show()

# rolling mean
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

# adjusting the size of matplotlib
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# adjusting the style of matplotlib
style.use('ggplot')

# moving average
close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
plt.show()
# plotting return rate
# rets = close_px / close_px.shift(1) - 1
# rets.plot(label='return')
# plt.show()

# Stock returns rate and risk
# plt.scatter(retscomp.mean(), retscomp.std())
# plt.xlabel('Expected returns')
# plt.ylabel('Risk')
# for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
# plt.annotate(
#    label,
#    xy=(x, y), xytext=(20, -20),
#    textcoords='offset points', ha='right', va='bottom',
#    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


# def best_fit_slope_and_intercept(mean, std):
# m = (((mean(mean)*mean(std)) - mean(mean*std)) /
# ((mean(mean)*mean(mean)) - mean(mean*mean)))

# b = mean(std) - m*mean(mean)
# return m, b


# m, b = best_fit_slope_and_intercept(mean, std)
# print(m, b)
# 33regression_line = [(m*x)+b for x in mean]
# 3regression_line = []
# 3for x in mean:
# regression_line.append((m*x)+b)
# 3style.use('ggplot')
# plt.scatter(mean, std, color='003f72')
# plt.plot(mean, regression_line)
# plt.show()
