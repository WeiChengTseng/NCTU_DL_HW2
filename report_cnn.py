import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import csv
plt.style.use('ggplot')

EPOCH = 200

def smoothing(curve, a=21):
    x = lambda a: max(a, 0)
    return np.array(list(map(x, signal.savgol_filter(curve, a, 2))))

# -----------------------------------------------------------------------------

plt.rcParams["figure.figsize"] = (8, 4.5)
f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel3_stride1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, 2 CNN layers')

f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel5_stride1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 5x5, 2 CNN layers')

f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel7_stride1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 7x7, 2 CNN layers')

f = csv.DictReader(
    open('./result/csv/run-SmallCNN_crop8_wd0.0001_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH+50], value[:EPOCH+50]
plt.plot(step, smoothing(value), label='filter 3x3, 4 CNN layers')

plt.yscale('log')
plt.xlabel('training time (seconds)')
plt.ylabel('error rate')
plt.title('CNN with Different Filter Size')
plt.legend()

plt.savefig('cnn_filter.png', dpi=400)
plt.close()

# -----------------------------------------------------------------------------

plt.rcParams["figure.figsize"] = (8, 4.5)
f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel3_stride1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, stride 1')

f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel3_stride2_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, stride 2')

f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel3_stride3_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, stride 3')

f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel3_stride4_dilation1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, stride 4')

# plt.yscale('log')
plt.xlabel('training time (seconds)')
plt.ylabel('error rate')
plt.title('CNN with Different Stride')
plt.legend()

plt.savefig('cnn_stride.png', dpi=400)
plt.close()


f = csv.DictReader(
    open('./result/csv/run-CNN_exp_kernel3_stride1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, 2 CNN layer')

f = csv.DictReader(
    open('./result/csv/run-CNN3_exp_kernel3_stride1_dilation1_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH], value[:EPOCH]
plt.plot(step, smoothing(value), label='filter 3x3, 3 CNN layer')

f = csv.DictReader(
    open('./result/csv/run-SmallCNN_crop8_wd0.0001_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH+50], value[:EPOCH+50]
plt.plot(step, smoothing(value), label='filter 3x3, 4 CNN layers')

f = csv.DictReader(
    open('./result/csv/run-SmallCNN5_crop8_wd0.0001_dropout_test-tag-accuracy.csv',
         'r'))
step, value = zip(*[(float(r['Wall time']), 1-float(r['Value'])) for r in f])
step = np.array(step) - step[0]
step, value = step[:EPOCH-80], value[:EPOCH-80]
plt.plot(step, smoothing(value), label='filter 3x3, 5 CNN layers')


plt.xlabel('training time (seconds)')
plt.ylabel('error rate')
plt.title('CNN with Different Stride')
plt.legend()

plt.savefig('cnn_layer.png', dpi=400)
plt.close()