import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

f = csv.DictReader(
    open(
        './result/csv/run-lstm_bs5_hidden10_embed10_lrdc1_test-tag-accuracy.csv',
        'r'))

for row in f:
    print(row)