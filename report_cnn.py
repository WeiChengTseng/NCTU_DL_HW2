import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import csv
plt.style.use('ggplot')


def smoothing(curve, a=51):
    x = lambda a: max(a, 0)
    return np.array(list(map(x, signal.savgol_filter(curve, a, 3))))


plt.rcParams["figure.figsize"] = (8, 4.5)
f = csv.DictReader(
    open('./result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_test-tag-loss.csv',
         'r'))
step, value = zip(*[(int(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
plt.plot(step, (value), label='test')

f = csv.DictReader(
    open('./result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_train-tag-loss.csv',
         'r'))
step, value = zip(*[(int(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
plt.plot(step, (value), label='train')

plt.xlabel('training time')
plt.ylabel('Error Rate')
plt.title('CNN with Different Filter size')
plt.legend()

plt.savefig('cnn_filter.png', dpi=400)
plt.close()