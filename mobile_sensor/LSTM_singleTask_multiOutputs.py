"""
Keras LSTM, single task & multi-label prediction

"""

import numpy as np
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN,Convolution1D,MaxPooling1D,Flatten,Convolution2D
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import Input, Dense
from numpy.random import seed
import os
import warnings
from tensorflow import set_random_seed
import random as rn


seed(42)
rn.seed(12345)
set_random_seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_dataset(datasource: str) -> (np.ndarray, MinMaxScaler):
    """
    The function loads dataset from given file name and uses MinMaxScaler to transform data
    :param datasource: file name of data source
    :return: tuple of dataset and the used MinMaxScaler
    """

    # load the dataset
    dataframe = read_csv(datasource, index_col=0)
    dataframe = dataframe.drop('label_source', axis=1)  # drop the last column

    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.iloc[0:3000]  # first 3000 rows of dataframe

    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler

#take one input dataset and split it into train and test
def split_dataset(dataset, scaled, look_back, n_columns,n_labels,ratio):

    # frame as supervised learning
    reframed = series_to_supervised(scaled, look_back, 1)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train_data = int(len(dataset) * ratio)
    train = values[:n_train_data, :]
    test = values[n_train_data:, :]
    print ('test data-----:')
    print (test[0:5,:])
    # split into input and outputs
    n_obs = look_back * n_columns
    train_X, train_y = train[:, :n_obs], train[:, -n_labels:]  # labels are the last 6 columns
    test_X, test_y = test[:, :n_obs], test[:, -n_labels:]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], look_back, n_columns))
    test_X = test_X.reshape((test_X.shape[0], look_back, n_columns))

    return train_X, train_y, test_X, test_y

# def split_dataset(scaled, look_back, n_columns,n_labels):
#
#     # frame as supervised learning
#     reframed = series_to_supervised(scaled, look_back, 1)
#
#     # split into train and test sets
#     values = reframed.values
#
#     # split into input and outputs
#     n_obs = look_back * n_columns
#     data_X, data_y = values[:, :n_obs], values[:, -n_labels:]  # labels are the last 51 columns
#     print(data_X.shape, len(data_X), data_y.shape)
#     # reshape input to be 3D [samples, timesteps, features]
#     data_X = data_X.reshape((data_X.shape[0], look_back, n_columns))
#
#     return data_X, data_y

def build_model(train):
    """
    Model 1: keras Sequential model
    """
    # model = Sequential()
    # model.add(Convolution1D(nb_filter=128, filter_length=4, activation='relu',input_shape=(train.shape[1], train.shape[2])))
    # model.add(Convolution1D(nb_filter=64, filter_length=2, activation='relu'))
    # # model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(64, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(51, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

    """
    Model 2: keras Function model
    """
    inputs = Input(shape=(train.shape[1], train.shape[2]))
    x = Convolution1D(nb_filter=128, filter_length=4, activation='relu', input_shape=(train.shape[1], train.shape[2]))(inputs)
    x = Convolution1D(nb_filter=64, filter_length=2, activation='relu')(x)
    x = LSTM(64, activation='relu',dropout=0.2)(x) # 'return_sequences=True' if connectted to another LSTM or Conv layer.
    x = Dense(64, activation='relu')(x)
    outputs = Dense(51,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

    return model


def prediction(model, test_X, test_y, timestamps, n_columns, n_labels, scaler):

    y_pred = model.predict(test_X)
    print(y_pred.shape)
    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-n_labels],y_pred), axis=1)
    print('before')
    print(y_predict.shape)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -n_labels:]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), n_labels))
    y_true = concatenate(( test_X[:, -n_columns:-n_labels],test_y ), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -n_labels:]

    print ("------Predicted labels: ---------")
    print (y_predict)
    print ("------True labels: ---------")
    print (y_true)
    
    return y_predict, y_true

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def BalanceAcc(y_pred,y_true):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            if y_pred[i][j] == 1 and y_true[i][j] == 1:
                tp += 1
            elif y_pred[i][j] == 1 and y_true[i][j] == 0:
                fp += 1
            elif y_pred[i][j] == 0 and y_true[i][j] == 0:
                tn += 1
            elif y_pred[i][j] == 0 and y_true[i][j] == 1:
                fn += 1
            total += 1
    # true positive rate
    sensitivity = float(tp) / (tp + fn)
    # true negative rate
    specificity = float(tn) / (tn + fp)
    # naive accuracy
    accuracy = float(tn + tp) / total
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;

    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print("-" * 10);

def plot(y_true, y_predict):

    labels = ['1', '2', '3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
              '21','22','23','24','25','26','27','28','29','30']
    for i in range (len(labels)):
        plt.figure(figsize=(24, 8))
        plt.plot(y_true[:,i], c='g', label='Actual')
        plt.plot(y_predict[:,i], c='r',  label='Predicted')
        plt.legend(fontsize='small')
        plt.title('Actual and Predicted ' + labels[i])
        plt.savefig('results/predicted_and_actural_'+labels[i] +'.eps', format="eps", dpi=200)


def main():
    dataset1 = 'data_csv/train/10users/74B86067-5D4B-43CF-82CF-341B76BEA0F4.features_labels.csv'

    dataset1, scaled1, scaler1 = load_dataset(dataset1)


    look_back = 20 # number of previous timestamp used for training
    n_columns = 276 # total columns
    n_labels = 51 # number of labels
    split_ratio = 0.8 # train & test data split ratio

    # split into train and test sets
    train_X, train_y, test_X, test_y = split_dataset(dataset1, scaled1, look_back, n_columns, n_labels, split_ratio)


    model = build_model(train_X)

    import time
    start_time = time.time()

    # fit network
    history = model.fit(train_X, train_y, epochs=40, batch_size=72,
                        # validation_data=(test_X, test_y),
                        validation_split=0.2,
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
                                                          mode='min')]
                        )

    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make a prediction
    y_predict, y_true = prediction(model, test_X, test_y, look_back, n_columns, n_labels, scaler1)

    # Round labels of the array to the nearest integer.
    y_predict = np.rint(y_predict)
    y_true = np.rint(y_true)

    y_predict[y_predict <= 0] = 0
    y_true[y_true <= 0] = 0

    # y_predict[y_predict >= 1] = 1
    # y_true[y_true >= 1] = 1

    # balance accuracy
    BalanceAcc(y_predict, y_true)


if __name__ == '__main__':
    main()
