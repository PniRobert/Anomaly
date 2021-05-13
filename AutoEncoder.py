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
from keras.models import Sequential
from keras import regularizers
from keras.utils import plot_model

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

def data_from_csv(filename, block_size):
    indexColName = "TimeGenerated [UTC]"
    firstFile = True
    custom_date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y, %I:%M:%S.%f %p")
    dataset = pd.read_csv(filename, sep=",", quotechar='"', doublequote=True,
            parse_dates=[indexColName], date_parser=custom_date_parser)
    dataset.set_index(indexColName)
    arr = dataset[["erro_rate", "avg_duration"]].to_numpy()
    originalShape = arr.shape
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(arr)
    return temp.reshape(originalShape[0] // block_size, block_size, originalShape[1])


total = 12 * 24 * 3
epochs = 1000
batch_size = 36

training = data_from_csv("data/training/051105132021.csv", batch_size)

# nn = autoencoder_model(training.shape[-2:])
nn = sequential_model(training.shape[-2:])
nn.compile(loss="mae", optimizer="adam")
plot_model(nn, show_shapes=True, to_file="mode_architecture.png")
history = nn.fit(training, training, batch_size=batch_size, epochs=epochs, validation_split=0.05, shuffle=True)
# plot_train_history(history, "train")

nn.save("test_model")



