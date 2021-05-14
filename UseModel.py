from datetime import datetime
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

indexColName = "TimeGenerated [UTC]"
error_rate_column = "erro_rate"
avg_column = "avg_duration"
data_column = [error_rate_column, avg_column]

def plot_loss(ds_result, column_to_display):
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    # sns.distplot(scored[column_to_display], bins = 20, kde = True, color = "blue")
    sns.histplot(scored[column_to_display], bins = 20, kde = True, color = "blue")
    plt.xlim([0.0,.5])
    plt.show()

def data_from_csv(filename, block_size):
    firstFile = True
    custom_date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y, %I:%M:%S.%f %p")
    dataset = pd.read_csv(filename, sep=",", quotechar='"', doublequote=True,
            parse_dates=[indexColName], date_parser=custom_date_parser)
    dataset.set_index(indexColName)
    arr = dataset[data_column].to_numpy()
    originalShape = arr.shape
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(arr)
    return (temp.reshape(originalShape[0] // block_size, block_size, originalShape[1]), dataset)

sns.set(color_codes=True, rc={'figure.figsize':(11, 4)})
batch_size = 36
loss_column_name = "loss_mae"
threshold_column_name = "threshold"
anomaly_column_name = "anomaly"
THRESHOLD=0.225

(test, ds) = data_from_csv("data/test/query_data.csv", batch_size)

nn = load_model("test_model")
# nn.summary()
temp = nn.predict(test)
pred = temp.reshape(temp.shape[0] * temp.shape[1],  temp.shape[2])
scored = pd.DataFrame(index=ds.index)
test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])
scored[loss_column_name] = np.mean(np.abs(pred - test), axis=1)
# plot_loss(scored, loss_column_name)
scored[threshold_column_name] = THRESHOLD
scored[anomaly_column_name] = scored[loss_column_name] > scored[threshold_column_name]
scored[error_rate_column] = ds[error_rate_column]
scored[avg_column] = ds[avg_column]
scored[indexColName] = ds[indexColName]
scored.set_index(indexColName)

anomaly = scored[scored[anomaly_column_name] == True]
with pd.option_context("display.max_rows", None, 'display.max_columns', None):
    print(anomaly[[indexColName, loss_column_name]])








