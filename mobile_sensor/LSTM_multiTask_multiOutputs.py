"""
Keras LSTM, multi-task & multi-outputs prediction (also can be used in multi-label situation)

"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

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
import os
import warnings
import glob



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


def build_model(data_list,
                task_num, lstm_layer, drop, r_drop, shared_layer,dense_num, n_labels):

    """
    Keras Function model
    """
    concate_list = []
    input_list = []
    for i in range(0, task_num):
        locals()['input' + str(i)] = Input(shape=(data_list[i].shape[1], data_list[i].shape[2]),
            name='input' + str(i))
        locals()['lstm_out' + str(i)] = LSTM(lstm_layer, activation='relu', dropout=drop, recurrent_dropout=r_drop)(
            locals()['input' + str(i)])
        concate_list.append(locals()['lstm_out' + str(i)])
        input_list.append(locals()['input' + str(i)])

    concate_layer = keras.layers.concatenate(concate_list)
    dense_shared = Dense(shared_layer, activation='relu')(concate_layer)

    output_list = []
    for i in range(0, task_num):
        locals()['sub' + str(i)] = Dense(dense_num, activation='relu')(dense_shared)
        locals()['out' + str(i)] = Dense(n_labels, activation='sigmoid')(locals()['sub' + str(i)])
        output_list.append(locals()['out' + str(i)])

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    print (model.summary())

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

    # Round labels of the array to the nearest integer.
    y_predict = np.rint(y_predict)
    y_true = np.rint(y_true)

    y_predict[y_predict <= 0] = 0
    y_true[y_true <= 0] = 0

    Bacc = BalanceAcc(y_predict,y_true)

    return Bacc


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

    # # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    #
    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print("-" * 10);

    return accuracy, sensitivity, specificity, balanced_accuracy


def main():
    # network parameters
    task_num = 6
    lstm_layer = 64
    drop = 0.2
    r_drop = 0.2
    shared_layer = 576
    dense_num = 128

    look_back = 20  # number of previous timestamp used for training
    n_columns = 276  # total columns
    n_labels = 51  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    trainX_list = []
    trainy_list = []
    testX_list = []
    testy_list = []
    file_list = glob.glob('data_csv/train/5users/*.csv')
    # print (file_list[0])

    for i in range(len(file_list)):
        locals()['dataset' + str(i)] = file_list[i]
        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = load_dataset(
            locals()['dataset' + str(i)])
        locals()['train_X' + str(i)], locals()['train_y' + str(i)], locals()['test_X' + str(i)], locals()[
            'test_y' + str(i)] = split_dataset(locals()['dataset' + str(i)], locals()['scaled' + str(i)], look_back,
                                               n_columns, n_labels, split_ratio)
        trainX_list.append(locals()['train_X' + str(i)])
        trainy_list.append(locals()['train_y' + str(i)])
        testX_list.append(locals()['test_X' + str(i)])
        testy_list.append(locals()['test_y' + str(i)])


    model = build_model(trainX_list,task_num, lstm_layer, drop, r_drop, shared_layer, dense_num, n_labels)

    import time
    start_time = time.time()

    # fit network
    history = model.fit(trainX_list, trainy_list,
                        epochs=1,
                        batch_size=72,
                        validation_split=0.25,
                        # validation_data=(testX_list, testy_list),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
                                                          mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # make prediction
    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6 = model.predict(testX_list)

    # write parameters & results to file
    file = open('results/Baseline_results(6)2.txt', 'w')

    file.write('task_num:' + str(task_num) + '\n')
    file.write('lstm_layer:' + str(lstm_layer) + '\n')
    file.write('drop:' + str(drop) + '\n')
    file.write('r_drop:' + str(r_drop) + '\n')
    file.write('shared_layer:' + str(shared_layer) + '\n')
    file.write('dense_num:' + str(dense_num) + '\n')
    file.write('running time:' + str(end_time - start_time) + '\n')


    sum_bacc = 0
    sum_TPR = 0

    # balance accuracy
    for i in range(len(file_list)):
        locals()['Bacc' + str(i)] = evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], locals()['y_pred' + str(i+1)],
                                               look_back, n_columns, n_labels, locals()['scaler' + str(i)])
        sum_bacc = sum_bacc + (locals()['Bacc' + str(i)])[3]
        sum_TPR = sum_TPR + (locals()['Bacc' + str(i)])[1]

        file.write ('Accuracy:'+' ' + str((locals()['Bacc' + str(i)])[0])+' ')
        file.write ('TPR:'+' ' + str((locals()['Bacc' + str(i)])[1])+' ')
        file.write ('TNR:'+' '+ str((locals()['Bacc' + str(i)])[2])+' ')
        file.write ('Bacc:'+' ' + str((locals()['Bacc' + str(i)])[3])+ '\n')

    file.write ('sum_bacc: ' + str(sum_bacc) +'\n')
    file.write ('sum_TPR: ' + str(sum_TPR)+'\n')

if __name__ == '__main__':
    main()


# back-up code

# def build_model(data1,data2,data3,data4,data5,data6,data7,data8,data9):
#
#     """
#     Keras Function model
#     """
#     input1 = Input(shape=(data1.shape[1], data1.shape[2]), name='input1')
#     lstm_out1 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(input1)
#     input2 = Input(shape=(data2.shape[1], data2.shape[2]), name='input2')
#     lstm_out2 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input2)
#     input3 = Input(shape=(data3.shape[1], data3.shape[2]), name='input3')
#     lstm_out3 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input3)
#     input4 = Input(shape=(data4.shape[1], data4.shape[2]), name='input4')
#     lstm_out4 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input4)
#     input5 = Input(shape=(data5.shape[1], data5.shape[2]), name='input5')
#     lstm_out5 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input5)
#     input6 = Input(shape=(data6.shape[1], data6.shape[2]), name='input6')
#     lstm_out6 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input6)
#     input7 = Input(shape=(data7.shape[1], data7.shape[2]), name='input7')
#     lstm_out7 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input7)
#     input8 = Input(shape=(data8.shape[1], data8.shape[2]), name='input8')
#     lstm_out8 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input8)
#     input9 = Input(shape=(data9.shape[1], data9.shape[2]), name='input9')
#     lstm_out9 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input9)
#
#     concate_layer = keras.layers.concatenate([lstm_out1, lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8,lstm_out9])
#     x = Dense(576, activation='relu')(concate_layer)
#     x = Dense(576, activation='relu')(x)
#
#     sub1 = Dense(64,activation='relu')(x)
#     sub2 = Dense(64,activation='relu')(x)
#     sub3 = Dense(64,activation='relu')(x)
#     sub4 = Dense(64,activation='relu')(x)
#     sub5 = Dense(64,activation='relu')(x)
#     sub6 = Dense(64,activation='relu')(x)
#     sub7 = Dense(64,activation='relu')(x)
#     sub8 = Dense(64,activation='relu')(x)
#     sub9 = Dense(64,activation='relu')(x)
#     out1 = Dense(51, activation='sigmoid')(sub1)
#     out2 = Dense(51, activation='sigmoid')(sub2)
#     out3 = Dense(51, activation='sigmoid')(sub3)
#     out4 = Dense(51, activation='sigmoid')(sub4)
#     out5 = Dense(51, activation='sigmoid')(sub5)
#     out6 = Dense(51, activation='sigmoid')(sub6)
#     out7 = Dense(51, activation='sigmoid')(sub7)
#     out8 = Dense(51, activation='sigmoid')(sub8)
#     out9 = Dense(51, activation='sigmoid')(sub9)
#
#     model = Model(inputs=[input1,input2,input3,input4,input5,input6,input7,input8,input9],
#                   outputs=[out1,out2,out3,out4,out5,out6,out7,out8,out9])
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
#
#     return model




    # dataset1 = 'data_csv/train/10users/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv'
    # dataset2 = 'data_csv/train/10users/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E.features_labels.csv'
    # dataset3 = 'data_csv/train/10users/1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842.features_labels.csv'
    # dataset4 = 'data_csv/train/10users/2C32C23E-E30C-498A-8DD2-0EFB9150A02E.features_labels.csv'
    # dataset5 = 'data_csv/train/10users/4FC32141-E888-4BFF-8804-12559A491D8C.features_labels.csv'
    # dataset6 = 'data_csv/train/10users/5EF64122-B513-46AE-BCF1-E62AAC285D2C.features_labels.csv'
    # dataset7 = 'data_csv/train/10users/7CE37510-56D0-4120-A1CF-0E23351428D2.features_labels.csv'
    # dataset8 = 'data_csv/train/10users/11B5EC4D-4133-4289-B475-4E737182A406.features_labels.csv'
    # dataset9 = 'data_csv/train/10users/74B86067-5D4B-43CF-82CF-341B76BEA0F4.features_labels.csv'
    #
    #
    # dataset1, scaled1, scaler1 = load_dataset(dataset1)
    # dataset2, scaled2, scaler2 = load_dataset(dataset2)
    # dataset3, scaled3, scaler3 = load_dataset(dataset3)
    # dataset4, scaled4, scaler4 = load_dataset(dataset4)
    # dataset5, scaled5, scaler5 = load_dataset(dataset5)
    # dataset6, scaled6, scaler6 = load_dataset(dataset6)
    # dataset7, scaled7, scaler7 = load_dataset(dataset7)
    # dataset8, scaled8, scaler8 = load_dataset(dataset8)
    # dataset9, scaled9, scaler9 = load_dataset(dataset9)
    #
    #
    # look_back = 20  # number of previous timestamp used for training
    # n_columns = 276  # total columns
    # n_labels = 51  # number of labels
    # split_ratio = 0.8 # train & test data split ratio
    #
    #
    # # get train and test sets
    # train_X1, train_y1, test_X1, test_y1 = split_dataset(dataset1, scaled1, look_back, n_columns, n_labels, split_ratio)
    # train_X2, train_y2, test_X2, test_y2 = split_dataset(dataset2, scaled2, look_back, n_columns, n_labels, split_ratio)
    # train_X3, train_y3, test_X3, test_y3 = split_dataset(dataset3, scaled3, look_back, n_columns, n_labels, split_ratio)
    # train_X4, train_y4, test_X4, test_y4 = split_dataset(dataset4, scaled4, look_back, n_columns, n_labels, split_ratio)
    # train_X5, train_y5, test_X5, test_y5 = split_dataset(dataset5, scaled5, look_back, n_columns, n_labels, split_ratio)
    # train_X6, train_y6, test_X6, test_y6 = split_dataset(dataset6, scaled6, look_back, n_columns, n_labels, split_ratio)
    # train_X7, train_y7, test_X7, test_y7 = split_dataset(dataset7, scaled7, look_back, n_columns, n_labels, split_ratio)
    # train_X8, train_y8, test_X8, test_y8 = split_dataset(dataset8, scaled8, look_back, n_columns, n_labels, split_ratio)
    # train_X9, train_y9, test_X9, test_y9 = split_dataset(dataset9, scaled9, look_back, n_columns, n_labels, split_ratio)
    #
    #
    #
    # model = build_model(train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_X7,train_X8,train_X9)
    #
    # import time
    # start_time = time.time()
    #
    # # fit network
    # history = model.fit([train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_X7,train_X8,train_X9],
    #                     [train_y1,train_y2,train_y3,train_y4,train_y5,train_y6,train_y7,train_y8,train_y9],
    #                     epochs=50,
    #                     batch_size=72,
    #                     validation_data=([test_X1,test_X2,test_X3,test_X4,test_X5,test_X6,test_X7,test_X8,test_X9],
    #                                      [test_y1,test_y2,test_y3,test_y4,test_y5,test_y6,test_y7,test_y8,test_y9]),
    #                     verbose=2,
    #                     shuffle=False,
    #                     callbacks=[
    #                         keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
    #                                                       mode='min')]
    #                     )
    # end_time = time.time()
    # print('--- %s seconds ---' % (end_time - start_time))
    #
    # # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()
    #
    # # make prediction
    # y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8,y_pred9 = model.predict([test_X1,test_X2,test_X3,test_X4,test_X5,test_X6,test_X7,test_X8,test_X9])
    # # print (len(y_pred1))
    #
    #
    # # balance accuracy
    # Bacc1 = evaluation(test_X1, test_y1, y_pred1, look_back, n_columns, n_labels, scaler1)
    # Bacc2 = evaluation(test_X2, test_y2, y_pred2, look_back, n_columns, n_labels, scaler2)
    # Bacc3 = evaluation(test_X3, test_y3, y_pred3, look_back, n_columns, n_labels, scaler3)
    # Bacc4 = evaluation(test_X4, test_y4, y_pred4, look_back, n_columns, n_labels, scaler4)
    # Bacc5 = evaluation(test_X5, test_y5, y_pred5, look_back, n_columns, n_labels, scaler5)
    # Bacc6 = evaluation(test_X6, test_y6, y_pred6, look_back, n_columns, n_labels, scaler6)
    # Bacc7 = evaluation(test_X7, test_y7, y_pred7, look_back, n_columns, n_labels, scaler7)
    # Bacc8 = evaluation(test_X8, test_y8, y_pred8, look_back, n_columns, n_labels, scaler8)
    # Bacc9 = evaluation(test_X9, test_y9, y_pred9, look_back, n_columns, n_labels, scaler9)

