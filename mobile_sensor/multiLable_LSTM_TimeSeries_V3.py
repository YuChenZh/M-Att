import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from pandas import read_csv
import matplotlib.pyplot as plt
import os
import warnings
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

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
    dataframe = dataframe.drop('label_source', axis=1) # drop the last column

    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)

    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler

def split_dataset(scaled, look_back, n_columns,n_labels):

    # frame as supervised learning
    reframed = series_to_supervised(scaled, look_back, 1)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values

    # split into input and outputs
    n_obs = look_back * n_columns
    data_X, data_y = values[:, :n_obs], values[:, -n_labels:]  # labels are the last 51 columns
    print(data_X.shape, len(data_X), data_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    data_X = data_X.reshape((data_X.shape[0], look_back, n_columns))

    return data_X, data_y

def build_model(train):
    """
    The function builds a keras Sequential model
    :return: keras Sequential model
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(train.shape[1], train.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(51, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def prediction(model, test_X, test_y, timestamps, n_columns, n_labels, scaler):

    y_pred = model.predict(test_X)
    print(y_pred.shape)
    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-n_labels],y_pred), axis=1)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -n_labels:]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), n_labels))
    y_true = concatenate((test_X[:, -n_columns:-n_labels],test_y), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -n_labels:]

    print ("------Predicted labels: ---------")
    print (y_pred)
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

    # # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    #
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
        # plt.show()

def main():
    train_data = 'data_csv/train/train2.csv'
    test_data = 'data_csv/train/33A85C34-CFE4-4732-9E73-0A7AC861B27A.features_labels.csv'

    train_dataset, train_scaled,train_scaler = load_dataset(train_data)
    test_dataset, test_scaled,test_scaler = load_dataset(test_data)

    look_back = 20  # number of previous timestamp used for training
    n_columns = 276  # total columns
    n_labels = 51  # number of labels

    # split into train and test sets
    train_X, train_y= split_dataset(train_scaled, look_back, n_columns,n_labels)
    test_X, test_y= split_dataset(test_scaled, look_back, n_columns,n_labels)

    model = build_model(train_X)
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
                                                          mode='min')]
                        )
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    y_predict, y_true = prediction(model, test_X, test_y, look_back, n_columns, n_labels, test_scaler)

    # Round labels of the array to the nearest integer.
    y_predict= np.rint(y_predict)
    y_true= np.rint(y_true)

    y_predict[y_predict <= 0] = 0
    y_true[y_true <= 0] = 0

    # y_predict[y_predict >= 1] = 1
    # y_true[y_true >= 1] = 1

    # balance accuracy
    BalanceAcc(y_predict, y_true)
    # plot(y_true, y_predict)

if __name__ == '__main__':
    main()
