"""
Keras LSTM, single task & single-label (one label at a time) prediction

"""

import numpy as np
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import Input, Dense
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def load_dataset(datasource: str) -> (np.ndarray, MinMaxScaler):
    """
    The function loads dataset from given file name and uses MinMaxScaler to transform data
    :param datasource: file name of data source
    :return: tuple of dataset and the used MinMaxScaler
    """
    # load the dataset
    dataframe = read_csv(datasource)
    dataframe.set_index('utc_time',inplace = True)
    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)

    # get data with label 'PM2.5'
    dataframe.drop(['PM10', 'NO2', 'CO', 'O3', 'SO2'], axis=1, inplace=True)

    # # get data with label 'PM10'
    # dataframe.drop(['PM2.5', 'NO2', 'CO', 'O3', 'SO2'], axis=1, inplace=True)

    # # get data with label 'NO2'
    # dataframe.drop(['PM2.5', 'PM10', 'CO', 'O3', 'SO2'], axis=1, inplace=True)

    # # get data with label 'CO'
    # dataframe.drop(['PM2.5', 'PM10', 'NO2', 'O3', 'SO2'], axis=1, inplace=True)

    # # get data with label 'O3'
    # dataframe.drop(['PM2.5', 'PM10', 'NO2', 'CO', 'SO2'], axis=1, inplace=True)

    # # get data with label 'SO2'
    # dataframe.drop(['PM2.5', 'PM10', 'CO', 'O3', 'NO2'], axis=1, inplace=True)

    dataset = dataframe.values
    # integer encode direction
    encoder = LabelEncoder()
    dataset[:, 8] = encoder.fit_transform(dataset[:, 8])
    dataset[:, 7] = encoder.fit_transform(dataset[:, 7])

    dataset = dataset.astype('float32')

    print('dataset.shape')
    print(dataset.shape)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler

def split_dataset(scaled, look_back, n_columns):

    # frame as supervised learning
    reframed = series_to_supervised(scaled, look_back, 1)
    # print(reframed.head())

    # split into train and test sets
    values = reframed.values

    # split into input and outputs
    n_obs = look_back * n_columns
    data_X, data_y = values[:, :n_obs], values[:, -1]  # label is the last column
    print(data_X.shape, len(data_X), data_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    data_X = data_X.reshape((data_X.shape[0], look_back, n_columns))

    return data_X, data_y

def build_model(train):
    """
    Model 1: keras Sequential model
    """
    # model = Sequential()
    # model.add(LSTM(64, activation='relu', input_shape=(train.shape[1], train.shape[2])))
    # model.add(Dropout(0.1))
    # model.add(Dense(6, activation='sigmoid'))
    # model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])

    """
    Model 2: keras Function model
    """
    inputs = Input(shape=(train.shape[1], train.shape[2]))
    x = LSTM(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])

    return model


def prediction(model, test_X, test_y, timestamps, n_columns, scaler):

    y_pred = model.predict(test_X)
    print(y_pred.shape)
    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-1],y_pred), axis=1)
    print(y_predict.shape)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    y_true = concatenate(( test_X[:,-n_columns :-1],test_y ), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -1]

    # print ("------Predicted labels: ---------")
    # print (y_predict)
    # print ("------True labels: ---------")
    # print (y_true)


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


def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))

def plot(y_true, y_predict,Smape):

    # columns_predict = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    plt.figure(figsize=(24, 8))
    plt.plot(y_true, c='g', label='Actual')
    plt.plot(y_predict, c='r',  label='Predicted')
    plt.legend(fontsize='small')
    plt.title('Actual and Predicted ' + 'PM2.5_ ' + 'Smape:'+ Smape )
    plt.savefig('Results/SingleLabelModel_predicted_and_actural_'+'PM2.5'+'.eps', format="eps", dpi=300)
    plt.show()


def main():
    train_data = 'data/preprocessed_data/bj_merged.csv'
    test_data = 'data/preprocessed_data/test/bj_daxing_201805.csv'

    dataset_train, scaled_train, scaler_train = load_dataset(train_data)
    dataset_test, scaled_test, scaler_test = load_dataset(test_data)

    look_back = 20 # number of previous timestamp used for training
    n_columns = 10 # total columns
    split_ratio = 0.8 # train & test data split ratio

    # split into train and test sets
    train_X, train_y = split_dataset(scaled_train, look_back, n_columns)
    test_X, test_y = split_dataset(scaled_test, look_back, n_columns)


    model = build_model(train_X)
    # fit network
    history = model.fit(train_X, train_y, epochs=40, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2,
                                                          mode='min')]
                        )
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make a prediction
    y_predict, y_true = prediction(model, test_X, test_y, look_back, n_columns, scaler_test)
    Smape = smape(y_true, y_predict)

    print('Test Smape: %.3f' % Smape)

    Smape = '%.3f' % float(Smape)

    # plot(y_true, y_predict,Smape)

if __name__ == '__main__':
    main()
