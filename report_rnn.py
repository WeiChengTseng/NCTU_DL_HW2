import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import csv
plt.style.use('ggplot')


def smoothing(curve):
    return signal.savgol_filter(curve, 51, 3)


f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_test-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
plt.plot(step, smoothing(value), label='test')
f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_train-tag-accuracy_train.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
plt.plot(step, smoothing(value), label='train')
plt.xlabel('training step')
plt.ylabel('accuracy')
plt.title('LSTM Error Rate Curve')
plt.legend()
plt.show()