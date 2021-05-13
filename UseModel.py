from datetime import datetime
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

data_column = ["erro_rate", "avg_duration"]

def data_from_csv(filename, block_size):
    indexColName = "TimeGenerated [UTC]"
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

sns.set(color_codes=True)
batch_size = 36
loss_column_name = "loss_mae"
THRESHOLD =0.175

(test, ds) = data_from_csv("data/training/050105032021.csv", batch_size)

nn = load_model("test_model")
# nn.summary()
temp = nn.predict(test)
pred = temp.reshape(temp.shape[0] * temp.shape[1],  temp.shape[2])
ds_pred = pd.DataFrame(pred, columns=data_column)
ds_pred.index = ds.index
scored = pd.DataFrame(index=ds.index)
test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])
scored[loss_column_name] = np.mean(np.abs(pred - test), axis=1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored[loss_column_name], bins = 20, kde= True, color = 'blue')
plt.xlim([0.0,.5])
plt.show()





