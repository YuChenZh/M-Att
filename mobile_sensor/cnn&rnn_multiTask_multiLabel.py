"""
Keras CNN & LSTM, multi-task & multi-label prediction

CNN is for the feature selection

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
from keras.layers import Input, Embedding, LSTM, Dense,Convolution1D,MaxPooling1D
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
import os
import warnings
from keras import backend as K
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

def build_model(data1,data2,data3,data4,data5,
                task_num, con_layer1, con_layer1_filter, con_layer2, con_layer2_filter,
                lstm_layer, drop, r_drop, shared_layer,dense_num, n_labels):

    """
    Keras Function model
    """
    concate_list = []
    input_list = []
    for i in range(0, task_num):
        locals()['input' + str(i)] = Input(shape=(locals()['data' + str(i + 1)].shape[1], locals()['data' + str(i + 1)].shape[2]),
            name='input' + str(i))
        locals()['cnn_out' + str(i)] = Convolution1D(nb_filter=con_layer1, filter_length=con_layer1_filter,
                                                     activation='relu')(locals()['input' + str(i)])
        locals()['cnn_out' + str(i)] = Convolution1D(nb_filter=con_layer2, filter_length=con_layer2_filter,
                                                     activation='relu')(locals()['cnn_out' + str(i)])
        locals()['lstm_out' + str(i)] = LSTM(lstm_layer, activation='relu',dropout=drop, recurrent_dropout=r_drop)(locals()['cnn_out' + str(i)])
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

    return model




def mycrossentropy(y_true, y_pred, e=0.3):
    nb_classes = 51
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2


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
    task_num = 5
    con_layer1 = 128
    con_layer1_filter = 8
    con_layer2 = 64
    con_layer2_filter = 4
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
        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = load_dataset(locals()['dataset' + str(i)])
        locals()['train_X'+str(i)], locals()['train_y'+str(i)], locals()['test_X'+str(i)], locals()['test_y'+str(i)] = split_dataset(locals()['dataset' + str(i)], locals()['scaled' + str(i)], look_back, n_columns, n_labels, split_ratio)
        trainX_list.append(locals()['train_X'+str(i)])
        trainy_list.append(locals()['train_y'+str(i)])
        testX_list.append(locals()['test_X' + str(i)])
        testy_list.append(locals()['test_y' + str(i)])

    model = build_model(locals()['train_X0'], locals()['train_X1'], locals()['train_X2'], locals()['train_X3'],
                        locals()['train_X4'], task_num, con_layer1, con_layer1_filter, con_layer2, con_layer2_filter,
                        lstm_layer, drop, r_drop, shared_layer, dense_num, n_labels)

    import time
    start_time = time.time()

    # fit network
    history = model.fit(trainX_list, trainy_list,
                        epochs=50,
                        batch_size=72,
                        validation_split=0.25,
                        # validation_data=(testX_list,testy_list),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
                                                          mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # make prediction
    y_pred1,y_pred2,y_pred3,y_pred4,y_pred5 = model.predict(testX_list)
    # print (len(y_pred1))

    # write parameters & results to file
    file = open('results/cnn&lstm_results(5)1.txt', 'w')

    file.write('task_num:' + str(task_num) + '\n')
    file.write('con_layer1:' + str(con_layer1) + '\n')
    file.write('con_layer1_filter:' + str(con_layer1_filter) + '\n')
    file.write('con_layer2:' + str(con_layer2) + '\n')
    file.write('con_layer2_filter:' + str(con_layer2_filter) + '\n')
    file.write('lstm_layer:' + str(lstm_layer) + '\n')
    file.write('drop:' + str(drop) + '\n')
    file.write('r_drop:' + str(r_drop) + '\n')
    file.write('shared_layer:' + str(shared_layer) + '\n')
    file.write('dense_num:' + str(dense_num) + '\n')

    sum_bacc = 0
    sum_TPR = 0

    # balance accuracy
    for i in range(len(file_list)):
        locals()['Bacc' + str(i)] = evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], y_pred1, look_back, n_columns, n_labels, locals()['scaler' + str(i)])

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


# # code backup ----
# def build_model(data1,data2,data3,data4,data5,data6,data7,data8,data9):
#
#     """
#     Keras Function model
#     """
#     input1 = Input(shape=(data1.shape[1], data1.shape[2]))
#     cnn_out1 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input1)
#     cnn_out1 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out1)
#     lstm_out1 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out1)
#
#     input2 = Input(shape=(data2.shape[1], data2.shape[2]), name='input2')
#     cnn_out2 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input2)
#     cnn_out2 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out2)
#     lstm_out2 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out2)
#
#     input3 = Input(shape=(data3.shape[1], data3.shape[2]), name='input3')
#     cnn_out3 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input3)
#     cnn_out3 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out3)
#     lstm_out3 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out3)
#
#     input4 = Input(shape=(data4.shape[1], data4.shape[2]), name='input4')
#     cnn_out4 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input4)
#     cnn_out4 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out4)
#     lstm_out4 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out4)
#
#     input5 = Input(shape=(data5.shape[1], data5.shape[2]), name='input5')
#     cnn_out5 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input5)
#     cnn_out5 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out5)
#     lstm_out5 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out5)
#
#     input6 = Input(shape=(data6.shape[1], data6.shape[2]), name='input6')
#     cnn_out6 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input6)
#     cnn_out6 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out6)
#     lstm_out6 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out6)
#
#     input7 = Input(shape=(data7.shape[1], data7.shape[2]), name='input7')
#     cnn_out7 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input7)
#     cnn_out7 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out7)
#     lstm_out7 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out7)
#
#     input8 = Input(shape=(data8.shape[1], data8.shape[2]), name='input8')
#     cnn_out8 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input8)
#     cnn_out8 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out8)
#     lstm_out8 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out8)
#
#     input9 = Input(shape=(data9.shape[1], data9.shape[2]), name='input9')
#     cnn_out9 = Convolution1D(nb_filter=128, filter_length=8,activation='relu')(input9)
#     cnn_out9 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out9)
#     lstm_out9 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out9)
#
#     # concate_layer = keras.layers.concatenate([cnn_out1, cnn_out2,cnn_out3,cnn_out4,cnn_out5,cnn_out6,cnn_out7,cnn_out8,cnn_out9])
#
#     concate_layer = keras.layers.concatenate([lstm_out1, lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8,lstm_out9])
#     # x = LSTM(512, activation='relu')(concate_layer)
#     # x = LSTM(512, activation='relu')(x)
#
#     x = Dense(576, activation='relu')(concate_layer)
#     # x = Dense(576, activation='relu')(x)
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
#     model.compile(loss=mycrossentropy, optimizer='adam', metrics=['binary_accuracy'])
#
#     return model