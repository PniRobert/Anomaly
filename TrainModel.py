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

from datafetcher import indexColName, get_from_csv
from nn_model_util import sequential_model, build_loss
from data_visualizer import plot_train_history, plot_loss

loss_column_name = "loss_mae"
epochs = 1250
batch_size = 36

(training, ds_train) = get_from_csv("data/training", batch_size)

nn = sequential_model(training.shape[-2:])
nn.compile(loss="mae", optimizer="adam")
# plot_model(nn, show_shapes=True, to_file="mode_architecture.png")

history = nn.fit(training, training, batch_size=batch_size, epochs=epochs, validation_split=0.05, shuffle=True, verbose=False)
plot_train_history(history, "train", "loss", "val_loss")

test = training[0:5,:,:]
pred = nn.predict(test)
ds1 = build_loss(pred, test, loss_column_name, ds_train.index)
plot_loss(ds1, loss_column_name)

nn.save("improved_model")



