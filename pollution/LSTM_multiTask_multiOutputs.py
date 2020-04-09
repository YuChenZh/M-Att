"""
Keras LSTM, multi-task & multi-outputs prediction (also can be used in multi-label situation)

"""
import pandas as pd
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


def load_dataset(dataframe):
    """
    The function loads dataset from given file name and uses MinMaxScaler to transform data
    :param datasource: file name of data source
    :return: tuple of dataset and the used MinMaxScaler
    """
    # load the dataset
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


def build_model(data1,data2,data3,data4,data5,data6,data7,data8,data9):

    """
    Keras Function model
    """
    input1 = Input(shape=(data1.shape[1], data1.shape[2]), name='input1')
    lstm_out1 = LSTM(48, activation='relu',dropout=0.2, recurrent_dropout=0.2)(input1)
    input2 = Input(shape=(data2.shape[1], data2.shape[2]), name='input2')
    lstm_out2 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input2)
    input3 = Input(shape=(data3.shape[1], data3.shape[2]), name='input3')
    lstm_out3 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input3)
    input4 = Input(shape=(data4.shape[1], data4.shape[2]), name='input4')
    lstm_out4 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input4)
    input5 = Input(shape=(data5.shape[1], data5.shape[2]), name='input5')
    lstm_out5 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input5)
    input6 = Input(shape=(data6.shape[1], data6.shape[2]), name='input6')
    lstm_out6 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input6)
    input7 = Input(shape=(data7.shape[1], data7.shape[2]), name='input7')
    lstm_out7 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input7)
    input8 = Input(shape=(data8.shape[1], data8.shape[2]), name='input8')
    lstm_out8 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input8)
    input9 = Input(shape=(data9.shape[1], data9.shape[2]), name='input9')
    lstm_out9 = LSTM(48, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input9)

    concate_layer = keras.layers.concatenate([lstm_out1, lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8,lstm_out9])
    x = Dense(216, activation='relu')(concate_layer)
    x = Dense(216, activation='relu')(x)

    sub1 = Dense(48,activation='relu')(x)
    sub2 = Dense(48,activation='relu')(x)
    sub3 = Dense(48,activation='relu')(x)
    sub4 = Dense(48,activation='relu')(x)
    sub5 = Dense(48,activation='relu')(x)
    sub6 = Dense(48,activation='relu')(x)
    sub7 = Dense(48,activation='relu')(x)
    sub8 = Dense(48,activation='relu')(x)
    sub9 = Dense(48,activation='relu')(x)
    out1 = Dense(6, activation='sigmoid')(sub1)
    out2 = Dense(6, activation='sigmoid')(sub2)
    out3 = Dense(6, activation='sigmoid')(sub3)
    out4 = Dense(6, activation='sigmoid')(sub4)
    out5 = Dense(6, activation='sigmoid')(sub5)
    out6 = Dense(6, activation='sigmoid')(sub6)
    out7 = Dense(6, activation='sigmoid')(sub7)
    out8 = Dense(6, activation='sigmoid')(sub8)
    out9 = Dense(6, activation='sigmoid')(sub9)

    model = Model(inputs=[input1,input2,input3,input4,input5,input6,input7,input8,input9],
                  outputs=[out1,out2,out3,out4,out5,out6,out7,out8,out9])
    model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])

    return model



def evaluation(test_X, test_y, y_pred, timestamps, n_columns, n_labels, scaler):

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

    # print("------Predicted labels: ---------")
    # print(y_predict)
    # print("------True labels: ---------")
    # print(y_true)

    Smape_all = smape(y_true, y_predict)

    return Smape_all


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
                    dpi=150)
        plt.show()


def main():
    train_data1 = pd.read_csv('data/preprocessed_data/bj_daxing.csv')
    train_data2 = pd.read_csv('data/preprocessed_data/bj_fangshan.csv')
    train_data3 = pd.read_csv('data/preprocessed_data/bj_mentougou.csv')
    train_data4 = pd.read_csv('data/preprocessed_data/bj_tongzhou.csv')
    train_data5 = pd.read_csv('data/preprocessed_data/bj_shunyi.csv')
    train_data6 = pd.read_csv('data/preprocessed_data/bj_miyun.csv')
    train_data7 = pd.read_csv('data/preprocessed_data/bj_huairou.csv')
    train_data8 = pd.read_csv('data/preprocessed_data/bj_pinggu.csv')
    train_data9 = pd.read_csv('data/preprocessed_data/bj_changping.csv')

    # train_data_all = pd.concat([train_data1,train_data2,train_data3,train_data4,train_data5,train_data6,train_data7,train_data8,train_data9])


    test_data1 = pd.read_csv('data/preprocessed_data/test/bj_daxing_201805.csv')
    test_data2 = pd.read_csv('data/preprocessed_data/test/bj_fangshan_201805.csv')
    test_data3 = pd.read_csv('data/preprocessed_data/test/bj_mentougou_201805.csv')
    test_data4 = pd.read_csv('data/preprocessed_data/test/bj_tongzhou_201805.csv')
    test_data5 = pd.read_csv('data/preprocessed_data/test/bj_shunyi_201805.csv')
    test_data6 = pd.read_csv('data/preprocessed_data/test/bj_miyun_201805.csv')
    test_data7 = pd.read_csv('data/preprocessed_data/test/bj_huairou_201805.csv')
    test_data8 = pd.read_csv('data/preprocessed_data/test/bj_pinggu_201805.csv')
    test_data9 = pd.read_csv('data/preprocessed_data/test/bj_changping_201805.csv')

    # test_data_all = pd.concat([test_data1,test_data2,test_data3,test_data4,test_data5,test_data6,test_data7,test_data8,test_data9])


    # dataset_train, scaled_train, scaler_train = load_dataset(train_data_all)
    dataset_train1, scaled_train1, scaler_train1 = load_dataset(train_data1)
    dataset_train2, scaled_train2, scaler_train2 = load_dataset(train_data2)
    dataset_train3, scaled_train3, scaler_train3 = load_dataset(train_data3)
    dataset_train4, scaled_train4, scaler_train4 = load_dataset(train_data4)
    dataset_train5, scaled_train5, scaler_train5 = load_dataset(train_data5)
    dataset_train6, scaled_train6, scaler_train6 = load_dataset(train_data6)
    dataset_train7, scaled_train7, scaler_train7 = load_dataset(train_data7)
    dataset_train8, scaled_train8, scaler_train8 = load_dataset(train_data8)
    dataset_train9, scaled_train9, scaler_train9 = load_dataset(train_data9)

    # dataset_test, scaled_test, scaler_test = load_dataset(test_data_all)

    dataset_test1, scaled_test1, scaler_test1 = load_dataset(test_data1)
    dataset_test2, scaled_test2, scaler_test2 = load_dataset(test_data2)
    dataset_test3, scaled_test3, scaler_test3 = load_dataset(test_data3)
    dataset_test4, scaled_test4, scaler_test4 = load_dataset(test_data4)
    dataset_test5, scaled_test5, scaler_test5 = load_dataset(test_data5)
    dataset_test6, scaled_test6, scaler_test6 = load_dataset(test_data6)
    dataset_test7, scaled_test7, scaler_test7 = load_dataset(test_data7)
    dataset_test8, scaled_test8, scaler_test8 = load_dataset(test_data8)
    dataset_test9, scaled_test9, scaler_test9 = load_dataset(test_data9)

    look_back = 20  # number of previous timestamp used for training
    n_columns = 15  # total columns
    n_labels = 6  # number of labels

    # get train and test sets
    # train_X, train_y = split_dataset(scaled_train, look_back, n_columns, n_labels)
    train_X1, train_y1 = split_dataset(scaled_train1, look_back, n_columns, n_labels)
    train_X2, train_y2 = split_dataset(scaled_train2, look_back, n_columns, n_labels)
    train_X3, train_y3 = split_dataset(scaled_train3, look_back, n_columns, n_labels)
    train_X4, train_y4 = split_dataset(scaled_train4, look_back, n_columns, n_labels)
    train_X5, train_y5 = split_dataset(scaled_train5, look_back, n_columns, n_labels)
    train_X6, train_y6 = split_dataset(scaled_train6, look_back, n_columns, n_labels)
    train_X7, train_y7 = split_dataset(scaled_train7, look_back, n_columns, n_labels)
    train_X8, train_y8 = split_dataset(scaled_train8, look_back, n_columns, n_labels)
    train_X9, train_y9 = split_dataset(scaled_train9, look_back, n_columns, n_labels)

    # test_X, test_y = split_dataset(scaled_test, look_back, n_columns, n_labels)
    test_X1, test_y1 = split_dataset(scaled_test1, look_back, n_columns, n_labels)
    test_X2, test_y2 = split_dataset(scaled_test2, look_back, n_columns, n_labels)
    test_X3, test_y3 = split_dataset(scaled_test3, look_back, n_columns, n_labels)
    test_X4, test_y4 = split_dataset(scaled_test4, look_back, n_columns, n_labels)
    test_X5, test_y5 = split_dataset(scaled_test5, look_back, n_columns, n_labels)
    test_X6, test_y6 = split_dataset(scaled_test6, look_back, n_columns, n_labels)
    test_X7, test_y7 = split_dataset(scaled_test7, look_back, n_columns, n_labels)
    test_X8, test_y8 = split_dataset(scaled_test8, look_back, n_columns, n_labels)
    test_X9, test_y9 = split_dataset(scaled_test9, look_back, n_columns, n_labels)


    model = build_model(train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_X7,train_X8,train_X9)

    import time
    start_time = time.time()

    # fit network
    history = model.fit([train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_X7,train_X8,train_X9],
                        [train_y1,train_y2,train_y3,train_y4,train_y5,train_y6,train_y7,train_y8,train_y9],
                        epochs=40,
                        batch_size=80,
                        validation_data=([test_X1,test_X2,test_X3,test_X4,test_X5,test_X6,test_X7,test_X8,test_X9],
                                         [test_y1,test_y2,test_y3,test_y4,test_y5,test_y6,test_y7,test_y8,test_y9]),
                        verbose=2,
                        shuffle=False
                        # callbacks=[
                        #     keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2,
                        #                                   mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()

    # make prediction
    y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8,y_pred9 = model.predict([test_X1,test_X2,test_X3,test_X4,test_X5,test_X6,test_X7,test_X8,test_X9])
    print (len(y_pred1))

    # evaluation
    Smape_test1 = evaluation(test_X1, test_y1, y_pred1, look_back, n_columns, n_labels, scaler_test1)
    Smape_test2 = evaluation(test_X2, test_y2, y_pred2, look_back, n_columns, n_labels, scaler_test2)
    Smape_test3 = evaluation(test_X3, test_y3, y_pred3, look_back, n_columns, n_labels, scaler_test3)
    Smape_test4 = evaluation(test_X4, test_y4, y_pred4, look_back, n_columns, n_labels, scaler_test4)
    Smape_test5 = evaluation(test_X5, test_y5, y_pred5, look_back, n_columns, n_labels, scaler_test5)
    Smape_test6 = evaluation(test_X6, test_y6, y_pred6, look_back, n_columns, n_labels, scaler_test6)
    Smape_test7 = evaluation(test_X7, test_y7, y_pred7, look_back, n_columns, n_labels, scaler_test7)
    Smape_test8 = evaluation(test_X8, test_y8, y_pred8, look_back, n_columns, n_labels, scaler_test8)
    Smape_test9 = evaluation(test_X9, test_y9, y_pred9, look_back, n_columns, n_labels, scaler_test9)

    all = Smape_test1+Smape_test2+Smape_test3+Smape_test4+Smape_test5+Smape_test6+Smape_test7+Smape_test8+Smape_test9
    print ('Test Smape of daxing: %.3f' % Smape_test1)
    print ('Test Smape of fangshan: %.3f' % Smape_test2)
    print ('Test Smape of mentougou: %.3f' % Smape_test3)
    print ('Test Smape of tongzhou: %.3f' % Smape_test4)
    print ('Test Smape of shunyi: %.3f' % Smape_test5)
    print ('Test Smape of miyun: %.3f' % Smape_test6)
    print ('Test Smape of huairou: %.3f' % Smape_test7)
    print ('Test Smape of pinggu: %.3f' % Smape_test8)
    print ('Test Smape of changping: %.3f' % Smape_test9)
    print ('Test Smape of all: %.3f' % all)


    # plot(y_true, y_predict, Smape)


if __name__ == '__main__':
    main()
