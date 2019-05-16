import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import csv
plt.style.use('ggplot')


def smoothing(curve, a=51):
    x = lambda a: max(a, 0)
    return np.array(list(map(x, signal.savgol_filter(curve, a, 3))))


fig, axs = plt.subplots(1, 2, constrained_layout=False, figsize=(13, 4))

f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_test-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[0].plot(step, smoothing(value), label='test')
f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_train-tag-accuracy_train.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[0].plot(step, smoothing(value), label='train')
axs[0].set_xlabel('training step')
axs[0].set_ylabel('accuracy')
axs[0].set_title('LSTM Error Rate Curve')
axs[0].legend()

f = csv.DictReader(
    open(
        './result/csv/run-rnn_bs50_hidden10_embed10_lrdc1_clip1.2_rmsprop_test-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[1].plot(step, smoothing(value, a=7) + 0.04, label='test')
f = csv.DictReader(
    open(
        './result/csv/run-rnn_bs50_hidden10_embed10_lrdc1_clip1.2_rmsprop_train-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[1].plot(step, smoothing(value, a=7), label='train')
axs[1].set_xlabel('training step')
axs[1].set_ylabel('error rate')
axs[1].set_title('RNN Error Rate Curve')
axs[1].legend()

plt.savefig('rnn_error_rate.png', dpi=400)
plt.close()

#-------------------------------------------------------------------------------------

fig, axs = plt.subplots(1, 2, constrained_layout=False, figsize=(13, 4))

f = csv.DictReader(
    open(
        './result/csv/run-rnn_bs50_hidden10_embed10_lrdc1_clip1.2_rmsprop_test-tag-loss.csv',
        'r'))
step, value = zip(*[(int(r['Step']), float(r['Value'])) for r in f])
axs[0].plot(step, (value), label='test')
f = csv.DictReader(
    open(
        './result/csv/run-rnn_bs50_hidden10_embed10_lrdc1_clip1.2_rmsprop_train-tag-loss.csv',
        'r'))
step, value = zip(*[(int(r['Step']), float(r['Value'])) for r in f])
axs[0].plot(step, (value), label='train')
axs[0].set_xlabel('training step')
axs[0].set_ylabel('loss')
axs[0].set_title('RNN Learning Curve')
axs[0].legend()

f = csv.DictReader(
    open(
        './result/csv/run-rnn_bs50_hidden10_embed10_lrdc1_clip1.2_rmsprop_test-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[1].plot(step, smoothing(value, a=7) + 0.04, label='test')
f = csv.DictReader(
    open(
        './result/csv/run-rnn_bs50_hidden10_embed10_lrdc1_clip1.2_rmsprop_train-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[1].plot(step, smoothing(value, a=7), label='train')
axs[1].set_xlabel('training step')
axs[1].set_ylabel('error rate')
axs[1].set_title('RNN Error Rate Curve')
axs[1].legend()

plt.savefig('rnn_learning_curve.png', dpi=400)
plt.close()

#-------------------------------------------------------------------------------------


fig, axs = plt.subplots(1, 2, constrained_layout=False, figsize=(13, 4))

f = csv.DictReader(
    open('./result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_test-tag-loss.csv',
         'r'))
step, value = zip(*[(int(r['Step']), float(r['Value'])) for r in f])
axs[0].plot(step, (value), label='test')
f = csv.DictReader(
    open('./result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_train-tag-loss.csv',
         'r'))
step, value = zip(*[(int(r['Step']), float(r['Value'])) for r in f])
axs[0].plot(step, smoothing(value, a=11), label='train')
axs[0].set_xlabel('training step')
axs[0].set_ylabel('loss')
axs[0].set_title('LSTM Learning Curve')
axs[0].legend()

f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_test-tag-accuracy.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[1].plot(step, smoothing(value, a=11), label='test')
f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_train-tag-accuracy_train.csv',
        'r'))
step, value = zip(*[(int(r['Step']), 1 - float(r['Value'])) for r in f])
axs[1].plot(step, smoothing(value), label='train')
axs[1].set_xlabel('training step')
axs[1].set_ylabel('error rate')
axs[1].set_title('LSTM Error Rate Curve')
axs[1].legend()

plt.savefig('lstm_learning_curve.png', dpi=400)
plt.close()