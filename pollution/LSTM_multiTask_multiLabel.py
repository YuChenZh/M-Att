"""
Keras LSTM, multi-task & multi-label prediction

"""
import numpy as np
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
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
    dataframe.set_index('utc_time', inplace=True)
    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)

    dataset = dataframe.values
    # integer encode direction
    encoder = LabelEncoder()
    dataset[:, 8] = encoder.fit_transform(dataset[:, 8])
    dataset[:, 7] = encoder.fit_transform(dataset[:, 7])

    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler


def split_dataset(scaled, look_back, n_columns,n_labels):

    # frame as supervised learning
    reframed = series_to_supervised(scaled, look_back, 1)
    # print(reframed.head())

    # split into train and test sets
    values = reframed.values

    # split into input and outputs
    n_obs = look_back * n_columns
    data_X, data_y = values[:, :n_obs], values[:, -n_labels:]  # labels are the last 51 columns
    print(data_X.shape, len(data_X), data_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    data_X = data_X.reshape((data_X.shape[0], look_back, n_columns))

    return data_X, data_y


def build_model(data1,data2):

    """
    Keras Function model
    """
    main_input = Input(shape=(data1.shape[1], data1.shape[2]), name='main_input')
    # x = Embedding(output_dim=512, input_dim=data1.shape[2], input_length=len(data1))(main_input)
    lstm_out1 = LSTM(64, activation='relu')(main_input)
    auxiliary_output = Dense(6, activation='sigmoid', name='aux_output')(lstm_out1)

    auxiliary_input = Input(shape=(data2.shape[1], data2.shape[2]), name='aux_input')
    lstm_out2 = LSTM(64, activation='relu')(auxiliary_input)
    x = keras.layers.concatenate([lstm_out1, lstm_out2])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(6, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)
    model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])

    return model

# def build_model(data,data1,data2,data3,data4,data5,data6,data7,data8):
#
#     """
#     Keras Function model
#     """
#     main_input = Input(shape=(data.shape[1], data.shape[2]), name='main_input')
#     # x = Embedding(output_dim=512, input_dim=data1.shape[2], input_length=len(data1))(main_input)
#     lstm_out1 = LSTM(64, activation='relu')(main_input)
#     auxiliary_output = Dense(6, activation='sigmoid', name='aux_output')(lstm_out1)
#
#     auxiliary_input = Input(shape=(data1.shape[1], data1.shape[2]), name='aux_input')
#     lstm_out2 = LSTM(64, activation='relu')(auxiliary_input)
#     auxiliary_input2 = Input(shape=(data2.shape[1], data2.shape[2]), name='aux_input2')
#     lstm_out3 = LSTM(64, activation='relu')(auxiliary_input2)
#     auxiliary_input3 = Input(shape=(data3.shape[1], data3.shape[2]), name='aux_input3')
#     lstm_out4 = LSTM(64, activation='relu')(auxiliary_input3)
#     auxiliary_input4 = Input(shape=(data4.shape[1], data4.shape[2]), name='aux_input4')
#     lstm_out5 = LSTM(64, activation='relu')(auxiliary_input4)
#     auxiliary_input5 = Input(shape=(data5.shape[1], data5.shape[2]), name='aux_input5')
#     lstm_out6 = LSTM(64, activation='relu')(auxiliary_input5)
#     auxiliary_input6 = Input(shape=(data6.shape[1], data6.shape[2]), name='aux_input6')
#     lstm_out7 = LSTM(64, activation='relu')(auxiliary_input6)
#     auxiliary_input7 = Input(shape=(data7.shape[1], data7.shape[2]), name='aux_input7')
#     lstm_out8 = LSTM(64, activation='relu')(auxiliary_input7)
#     auxiliary_input8 = Input(shape=(data8.shape[1], data8.shape[2]), name='aux_input8')
#     lstm_out9 = LSTM(64, activation='relu')(auxiliary_input8)
#
#
#     x = keras.layers.concatenate([lstm_out1, lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8,lstm_out9])
#     x = Dense(64, activation='relu')(x)
#     x = Dense(64, activation='relu')(x)
#     main_output = Dense(6, activation='sigmoid', name='main_output')(x)
#
#     model = Model(inputs=[main_input, auxiliary_input,auxiliary_input2],
#                   outputs=main_output)
#     rmsprop=keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-06)
#     model.compile(loss='mae', optimizer=rmsprop, metrics=['accuracy'])
#
#     return model


def prediction(model, test_X, test_X2, test_y, timestamps, n_columns, n_labels, scaler):
    y_pred = model.predict([test_X,test_X2])
    # print(y_pred.shape)
    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-n_labels], y_pred), axis=1)
    # print('before')
    # print(y_predict.shape)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -n_labels:]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), n_labels))
    y_true = concatenate((test_X[:, -n_columns:-n_labels], test_y), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -n_labels:]

    print("------Predicted labels: ---------")
    print(y_predict)
    print("------True labels: ---------")
    print(y_true)

    return y_predict, y_true


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
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


def plot(y_true, y_predict, Smape):
    columns_predict = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    for i in range(len(columns_predict)):
        plt.figure(figsize=(24, 8))
        plt.plot(y_true[:, i], c='g', label='Actual')
        plt.plot(y_predict[:, i], c='r', label='Predicted')
        plt.legend(fontsize='small')
        plt.title('Actual and Predicted ' + columns_predict[i] + '_Smape:' + Smape[i])
        plt.savefig('Results/Multilabel_MultiTask(8)_Model_predicted_and_actural_' + columns_predict[i] + '.eps', format="eps",
                    dpi=300)
        plt.show()


def main():
    train_data = 'data/preprocessed_data/bj_daxing.csv'
    train_data2 = 'data/preprocessed_data/daxing_grid_data280.csv'
    # train_data3 = 'data/preprocessed_data/daxing_grid_data259.csv'
    # train_data4 = 'data/preprocessed_data/bj_tongzhou.csv'
    # train_data5 = 'data/preprocessed_data/bj_shunyi.csv'
    # train_data6 = 'data/preprocessed_data/bj_miyun.csv'
    # train_data7 = 'data/preprocessed_data/bj_huairou.csv'
    # train_data8 = 'data/preprocessed_data/bj_pinggu.csv'
    # train_data9 = 'data/preprocessed_data/bj_changping.csv'

    test_data = 'data/preprocessed_data/test/bj_daxing_201805.csv'
    test_data2 = 'data/preprocessed_data/test/daxing_grid_data280_201805.csv'
    # test_data3 = 'data/preprocessed_data/test/daxing_grid_data259_201805.csv'
    # test_data4 = 'data/preprocessed_data/test/bj_tongzhou_201805.csv'
    # test_data5 = 'data/preprocessed_data/test/bj_shunyi_201805.csv'
    # test_data6 = 'data/preprocessed_data/test/bj_miyun_201805.csv'
    # test_data7 = 'data/preprocessed_data/test/bj_huairou_201805.csv'
    # test_data8 = 'data/preprocessed_data/test/bj_pinggu_201805.csv'
    # test_data9 = 'data/preprocessed_data/test/bj_changping_201805.csv'

    dataset_train, scaled_train, scaler_train = load_dataset(train_data)
    dataset_train2, scaled_train2, scaler_train2 = load_dataset(train_data2)
    # dataset_train3, scaled_train3, scaler_train3 = load_dataset(train_data3)
    # dataset_train4, scaled_train4, scaler_train4 = load_dataset(train_data4)
    # dataset_train5, scaled_train5, scaler_train5 = load_dataset(train_data5)
    # dataset_train6, scaled_train6, scaler_train6 = load_dataset(train_data6)
    # dataset_train7, scaled_train7, scaler_train7 = load_dataset(train_data7)
    # dataset_train8, scaled_train8, scaler_train8 = load_dataset(train_data8)
    # dataset_train9, scaled_train9, scaler_train9 = load_dataset(train_data9)

    dataset_test, scaled_test, scaler_test = load_dataset(test_data)
    dataset_test2, scaled_test2, scaler_test2 = load_dataset(test_data2)
    # dataset_test3, scaled_test3, scaler_test3 = load_dataset(test_data3)
    # dataset_test4, scaled_test4, scaler_test4 = load_dataset(test_data4)
    # dataset_test5, scaled_test5, scaler_test5 = load_dataset(test_data5)
    # dataset_test6, scaled_test6, scaler_test6 = load_dataset(test_data6)
    # dataset_test7, scaled_test7, scaler_test7 = load_dataset(test_data7)
    # dataset_test8, scaled_test8, scaler_test8 = load_dataset(test_data8)
    # dataset_test9, scaled_test9, scaler_test9 = load_dataset(test_data9)

    look_back = 20  # number of previous timestamp used for training
    n_columns = 15  # total columns
    n_labels = 6  # number of labels

    # get train and test sets
    train_X, train_y = split_dataset(scaled_train, look_back, n_columns, n_labels)
    train_X2, train_y2 = split_dataset(scaled_train2, look_back, n_columns, n_labels)
    # train_X3, train_y3 = split_dataset(scaled_train3, look_back, n_columns, n_labels)
    # train_X4, train_y4 = split_dataset(scaled_train4, look_back, n_columns, n_labels)
    # train_X5, train_y5 = split_dataset(scaled_train5, look_back, n_columns, n_labels)
    # train_X6, train_y6 = split_dataset(scaled_train6, look_back, n_columns, n_labels)
    # train_X7, train_y7 = split_dataset(scaled_train7, look_back, n_columns, n_labels)
    # train_X8, train_y8 = split_dataset(scaled_train8, look_back, n_columns, n_labels)
    # train_X9, train_y9 = split_dataset(scaled_train9, look_back, n_columns, n_labels)

    test_X, test_y = split_dataset(scaled_test, look_back, n_columns, n_labels)
    test_X2, test_y2 = split_dataset(scaled_test2, look_back, n_columns, n_labels)
    # test_X3, test_y3 = split_dataset(scaled_test3, look_back, n_columns, n_labels)
    # test_X4, test_y4 = split_dataset(scaled_test4, look_back, n_columns, n_labels)
    # test_X5, test_y5 = split_dataset(scaled_test5, look_back, n_columns, n_labels)
    # test_X6, test_y6 = split_dataset(scaled_test6, look_back, n_columns, n_labels)
    # test_X7, test_y7 = split_dataset(scaled_test7, look_back, n_columns, n_labels)
    # test_X8, test_y8 = split_dataset(scaled_test8, look_back, n_columns, n_labels)
    # test_X9, test_y9 = split_dataset(scaled_test9, look_back, n_columns, n_labels)

    # print (test_X.shape)
    # print (test_X2.shape)
    # print (train_X.shape)

    model = build_model(train_X,train_X2)

    # fit network
    history = model.fit([train_X, train_X2], train_y,
                        epochs=100,
                        batch_size=80,
                        validation_data=([test_X,test_X2], test_y),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=2,
                                                          mode='min')]
                        )

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()

    # make a prediction
    y_predict, y_true = prediction(model, test_X, test_X2, test_y, look_back, n_columns, n_labels, scaler_test)

    Smape_all = smape(y_true, y_predict)
    Smape_PM25 = smape(y_true[:, -6], y_predict[:, -6])
    Smape_PM10 = smape(y_true[:, -5], y_predict[:, -5])
    Smape_NO2 = smape(y_true[:, -4], y_predict[:, -4])
    Smape_CO = smape(y_true[:, -3], y_predict[:, -3])
    Smape_O3 = smape(y_true[:, -2], y_predict[:, -2])
    Smape_SO2 = smape(y_true[:, -1], y_predict[:, -1])

    Smape = [Smape_PM25, Smape_PM10, Smape_NO2, Smape_CO, Smape_O3, Smape_SO2]
    Smape = ["%.3f" % x for x in Smape]

    print('Test Smape of all: %.3f' % Smape_all)
    print('Test Smape of PM2.5: %.3f' % Smape_PM25)
    print('Test Smape of PM10: %.3f' % Smape_PM10)
    print('Test Smape of NO2: %.3f' % Smape_NO2)
    print('Test Smape of CO: %.3f' % Smape_CO)
    print('Test Smape of O3: %.3f' % Smape_O3)
    print('Test Smape of SO2: %.3f' % Smape_SO2)

    # plot(y_true, y_predict, Smape)


if __name__ == '__main__':
    main()
