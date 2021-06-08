import pandas as pd
import numpy as np
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.models import Sequential
from keras import regularizers

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

def build_loss(pred, actual, col_loss, dataset_index):
    pred1 = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
    actual1 = actual.reshape(actual.shape[0] * actual.shape[1], actual.shape[2])
    ds_loss =  pd.DataFrame()
    ds_loss[col_loss] = np.mean(np.abs(pred1 - actual1), axis=1)
    return ds_loss