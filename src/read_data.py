import pandas as pd
import numpy as np
import config


def read_data(datacsv_path:str):
    # config parameters
    y_multi_start = config.Y_MULTICLASS_INDEXES[0]
    y_multi_end = config.Y_MULTICLASS_INDEXES[1]
    y_single_start = config.Y_SINGLECLASS_INDEXES[0]
    y_single_end = config.Y_SINGLECLASS_INDEXES[1]
    x_start = config.X_STARTINDEX

    y_start = min(config.Y_MULTICLASS_INDEXES[0], config.Y_SINGLECLASS_INDEXES[0])

    # read data
    df = pd.read_csv(datacsv_path)
    data = np.array(df)
    m, n = data.shape

    # shuffle data
    np.random.shuffle(data)

    # separate data
        # dev
    data_dev = data[0:((int) ((1-config.TRAIN_PROP)*m))].T
    Y_dev = data_dev[y_multi_start:y_multi_end]
    X_dev = data_dev[x_start:n]
        # train
    data_train = data[((int) ((config.TRAIN_PROP)*m)):m].T
    Y_train = data_train[y_multi_start:y_multi_end]
    X_train = data_train[x_start:n]

    return Y_dev, X_dev, Y_train, X_train



"""
def read_data_(datacsv_path:str):
    df = pd.read_csv(datacsv_path)
    rawdata = np.array(df)
    m, n = rawdata.shape

    # shuffle data
    np.random.shuffle(rawdata)

    # creating array of size m and 6 times size n except for label string and fret
    data: np.ndarray = np.zeros((m, n*6))  

    for i in range(m):
        # add strings and frets labels
        stringslabel = eval(rawdata[i][0])
        fretslabel = eval(rawdata[i][1])
        for j,string in enumerate(stringslabel):
            data[i][(int) (string)-1] = 1
            data[i][(int) (string)-1 + 6] = (int) (fretslabel[j])

        # add pitches
        for j in range(2,n):
            pitches = eval(rawdata[i][j])
            for k in range(6):
                data[i][12+(j-2)*6+k] = pitches[k] if len(pitches)>k else 0
    
    data_dev = data[0:((int) ((1-config.TRAIN_PROP)*m))].T
    Y_dev = data_dev[0:6]
    X_dev = data_dev[12:n*6]

    data_train = data[((int) ((config.TRAIN_PROP)*m)):m].T
    Y_train = data_train[0:6]
    X_train = data_train[12:n*6]

    return Y_dev, X_dev, Y_train, X_train"""