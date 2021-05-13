import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import seed
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[0], X.shape[1]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[0])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[1]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

sns.set(color_codes=True)
traindata_folder = "data/training"
indexColName = "TimeGenerated [UTC]"
firstFile = True
custom_date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y, %I:%M:%S.%f %p")
for filename in os.listdir(traindata_folder):
    if firstFile:
        dataset = pd.read_csv(os.path.join(traindata_folder, filename), sep=",", quotechar='"', doublequote=True,
                     parse_dates=[indexColName], date_parser=custom_date_parser)
        dataset.set_index(indexColName)
        firstFile = False
    else:
        ds = pd.read_csv(os.path.join(traindata_folder, filename), sep=",", quotechar='"', doublequote=True,
                     parse_dates=[indexColName], date_parser=custom_date_parser)
        dataset.append(ds)

arr = dataset[["erro_rate", "avg_duration"]].to_numpy()
scaler = MinMaxScaler()
training = scaler.fit_transform(arr)
# print(training)

nn = autoencoder_model(training)
nn.compile()
nn.summary()
plot_model(nn, show_shapes=True, to_file="mode_architecture.png")


