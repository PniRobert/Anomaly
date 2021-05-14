import os
from datetime import datetime
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.utils import plot_model

indexColName = "TimeGenerated [UTC]"

# utility for train history
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

# utility for plot loss distribution
def plot_loss(ds_result, column_to_display):
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.histplot(ds_result[column_to_display], bins = 20, kde = True, color = "blue")
    plt.xlim([0.0,.5])
    plt.show()

# define the autoencoder network model
def autoencoder_model(inputshape):
    inputs = Input(shape=inputshape)
    L1 = LSTM(16, activation='relu', return_sequences=True,
               kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(2, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(inputshape[0])(L2)
    L4 = LSTM(2, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(inputshape[1]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

def sequential_model(inputshape):
    model = Sequential()
    model.add(LSTM(
        units=72,
        input_shape=inputshape
    ))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(n=inputshape[0]))
    model.add(LSTM(units=72, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(
        Dense(units=inputshape[1])))
    return model

# define get data from cvs file
def data_from_csv(data_folder, block_size):
    parts = []
    custom_date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y, %I:%M:%S.%f %p")
    for filename in os.listdir(data_folder):
        ds = pd.read_csv(os.path.join(data_folder, filename), sep=",", quotechar='"', doublequote=True,
                        parse_dates=[indexColName], date_parser=custom_date_parser)
        ds.set_index(indexColName)
        parts.append(ds)

    dataset = pd.concat(parts)
    arr = dataset[["erro_rate", "avg_duration"]].to_numpy()
    originalShape = arr.shape
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(arr)
    train_data = temp.reshape(originalShape[0] // block_size, block_size, originalShape[1])
    return (train_data, dataset)

# define build loss data set
def build_loss(pred, actual, col_loss, dataset_index):
    pred1 = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
    actual1 = actual.reshape(actual.shape[0] * actual.shape[1], actual.shape[2])
    ds_loss =  pd.DataFrame()
    ds_loss[col_loss] = np.mean(np.abs(pred1 - actual1), axis=1)
    return ds_loss

loss_column_name = "loss_mae"
epochs = 1250
batch_size = 36
sns.set(color_codes=True, rc={'figure.figsize':(11, 4)})

(training, ds_train) = data_from_csv("data/training", batch_size)

nn = sequential_model(training.shape[-2:])
nn.compile(loss="mae", optimizer="adam")
# plot_model(nn, show_shapes=True, to_file="mode_architecture.png")

history = nn.fit(training, training, batch_size=batch_size, epochs=epochs, validation_split=0.05, shuffle=True, verbose=False)
plot_train_history(history, "train")

test = training[0:5,:,:]
pred = nn.predict(test)
ds1 = build_loss(pred, test, loss_column_name, ds_train.index)
plot_loss(ds1, loss_column_name)
loss_column = ds1[loss_column_name]
mean = loss_column .mean()
sigma = loss_column .std()
threshold = mean + 1.96*sigma/math.sqrt(len(loss_column)) # 95%
print(f"Threshold: {threshold}")

nn.save("test_model")



