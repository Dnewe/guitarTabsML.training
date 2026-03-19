from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
import pandas as pd
import numpy as np

dataconfig = DataConfig()
modelconfig = ModelConfig()


def read_data(datacsv_path:str):
    # config parameters
    y_start = min(dataconfig.Y_STRINGS)
    y_end = max(dataconfig.Y_STRINGS)
    x_start = min(dataconfig.X_DATA)

    # read data
    df = pd.read_csv(datacsv_path)
    data = np.array(df)
    m, n = data.shape

    # shuffle data
    np.random.shuffle(data)
    print(data.shape)

    # separate data
    data_train = data[0 : modelconfig.TRAIN_SIZE]
    data_val = data[modelconfig.TRAIN_SIZE : modelconfig.TRAIN_SIZE+modelconfig.VAL_SIZE]
    data_test = data[modelconfig.TRAIN_SIZE+modelconfig.VAL_SIZE:]

    # separate features and labels
    X_train = data_train[:, x_start:]
    Y_train = data_train[:, :x_start]
    X_val = data_val[:, x_start:]
    Y_val = data_val[:, :x_start]
    X_test = data_test[:, x_start:]
    Y_test = data_test[:, :x_start]

    # normalize features
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - X_val.mean()) / X_val.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, train_mean, train_std
