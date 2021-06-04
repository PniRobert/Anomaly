import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

indexColName = "TimeGenerated [UTC]"
error_rate_column = "erro_rate"
avg_column = "avg_duration"
data_columns = [error_rate_column, avg_column]

def get_from_csv(data_folder, block_size):
    parts = []
    custom_date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y, %I:%M:%S.%f %p")
    for filename in os.listdir(data_folder):
        ds = pd.read_csv(os.path.join(data_folder, filename), sep=",", quotechar='"', doublequote=True,
                        parse_dates=[indexColName], date_parser=custom_date_parser)
        ds.set_index(indexColName)
        parts.append(ds)

    dataset = pd.concat(parts)
    arr = dataset[data_columns].to_numpy()
    original_shape = arr.shape
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(arr)
    train_data = temp.reshape(original_shape[0] // block_size, block_size, original_shape[1])
    return (train_data, dataset)