import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from pandas import read_csv
import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization


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

def tran_date_formate(row):
    return datetime.datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

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


if __name__ == '__main__':

    dataset = read_csv('data_csv/train/train2.csv', index_col=0)
    # dataset.drop('label_source')
    dataset = dataset.drop('label_source', axis=1)
    print (dataset.shape)

    # # For missing sensor-features, replace "nan" with the upper cell(ffill) or below cell(bfill) value
    # new_data = dataset.fillna(method='ffill')
    # # drop the feature column with all "nan"
    # new_data = new_data.fillna(method='bfill')
    dataset = dataset.fillna(0)  # after forward-filling and back-filling, fill the rest nan cells with 0

    # # plot feature trend (the first 8 features(columns))
    # plt.figure(figsize=(24, 8))
    # for i in range(8):
    #     plt.subplot(8, 1, i + 1)
    #     plt.plot(dataset.values[:, i])
    #     plt.title([i], y=0.5, loc='right')
    # plt.show()


    values = dataset.values  # convert to numpy array
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)


    n_features = 276 # total columns
    n_seconds = 20
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_seconds, 1)
    print(reframed.head())


    # split into train and test sets
    values = reframed.values
    n_train_data = int(len(dataset)*0.8)
    train = values[:n_train_data, :]
    test = values[n_train_data:, :]
    # split into input and outputs
    n_obs = n_seconds * n_features
    train_X, train_y = train[:, :n_obs], train[:, -51:] # labels are the last 51 columns
    test_X, test_y = test[:, :n_obs], test[:, -51:]
    print (test_y)
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_seconds, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_seconds, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    # design network
    model = Sequential()
    model.add(LSTM(60, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(51, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


    # make a prediction
    yhat = model.predict(test_X)
    print (yhat.shape)
    test_X = test_X.reshape((test_X.shape[0], n_seconds * n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -n_features:-51]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0:51]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 51))
    inv_y = concatenate((test_y, test_X[:, -n_features:-51]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0:51]

    # Round labels of the array to the nearest integer.
    inv_yhat= np.rint(inv_yhat)
    inv_y= np.rint(inv_y)

    inv_yhat[inv_yhat <= 0] = 0
    inv_y[inv_y <= 0] = 0
    # inv_yhat[inv_yhat >= 1] = 1
    # inv_y[inv_y >= 1] = 1


    print (type(inv_y))
    print (inv_y)
    print (inv_yhat)
    print (type(inv_yhat))

    np.savetxt("label.csv", inv_y, delimiter=",")

    BalanceAcc(inv_yhat, inv_y)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    # print (accuracy_score(inv_y, inv_yhat))

    # plt.figure(figsize=(24, 8))
    # train_predict = model.predict(train_X)
    # test_predict = model.predict(test_X)
    # plt.plot(values[:, -1], c='b')
    # plt.plot([x for x in train_predict], c='g')
    # plt.plot([None for _ in train_predict], c='y')
    # plt.plot([None for _ in train_predict] + [x for x in test_predict], c='r')
    # plt.show()

