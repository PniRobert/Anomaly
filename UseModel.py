from datetime import datetime
from datetime import timedelta
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import seed
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.models import load_model
import seaborn as sns
import datafetcher
from datafetcher import indexColName
from datafetcher import error_rate_column
from datafetcher import avg_column
import data_visualizer


batch_size = 36
loss_column_name = "loss_mae"
threshold_column_name = "threshold"
anomaly_column_name = "anomaly"
THRESHOLD=0.225

(test, ds) = datafetcher.get_from_csv("data/test", batch_size)

nn = load_model("test_model")
# nn.summary()
temp = nn.predict(test)
pred = temp.reshape(temp.shape[0] * temp.shape[1],  temp.shape[2])
scored = pd.DataFrame(index=ds.index)
test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])
scored[loss_column_name] = np.mean(np.abs(pred - test), axis=1)
# data_visualizer.plot_loss(scored, loss_column_name)
scored[threshold_column_name] = THRESHOLD
scored[anomaly_column_name] = scored[loss_column_name] > scored[threshold_column_name]
scored[error_rate_column] = ds[error_rate_column]
scored[avg_column] = ds[avg_column]
scored[indexColName] = ds[indexColName]
scored.set_index(indexColName)

anomaly = scored[scored[anomaly_column_name] == True]
with pd.option_context("display.max_rows", None, 'display.max_columns', None):
    print(anomaly[[indexColName, loss_column_name, avg_column, error_rate_column]])

print("       ")
print("       ")
timeFormat = "%Y-%m-%dT%H:%M:%S.000"
find = False
filters = []
ten_minute_delta = timedelta(minutes=30)
timeranges = anomaly[datafetcher.indexColName].squeeze().sort_values(ascending=True)
startTime = timeranges.iloc[0]
previousTime = startTime
for idx in range(1, timeranges.size -1):
    endTime = timeranges.iloc[idx]
    if endTime - previousTime > ten_minute_delta:
        if find:
            filters.append((startTime, previousTime))
            find = False
    else:
        if not find:
            find = True
            startTime = previousTime
    previousTime = endTime
if find:
    filters.append((startTime, endTime))

for p in filters:
    print(f"let startDateTime = datetime('{p[0].strftime(timeFormat)}');")
    print(f"let endDateTime = datetime('{p[1].strftime(timeFormat)}');")
    print("       ")
    print("       ")










