import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers.core import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Sequential



def features_labels_process(data):
    all_lables_list = [col for col in data if col.startswith('label')]
    all_features_list = [col for col in data if col not in all_lables_list]
    all_features_list.remove('timestamp')
    all_lables_list.remove('label_source')

    features = data[all_features_list]
    labels = data[all_lables_list]

    # For missing sensor-features, replace "nan" with the upper cell(ffill) or below cell(bfill) value
    features = features.fillna(method='ffill')
    # drop the feature column with all "nan"
    # features = features.dropna(axis=1, how='all')
    features = features.fillna(method='bfill')
    features = features.fillna(0) # after forward-filling and back-filling, fill the rest nan cells with 0

    # normalize feature vaules, by subtracting mean, then divide std
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(features)
    normalized_features = pd.DataFrame(np_scaled)

    # fill empty labels with 0
    labels = labels.fillna(0)

    return (normalized_features,labels)

def evaluation(y_pred,y_true):

    # Hamming loss
    hammingLoss = hamming_loss(y_true, y_pred)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    y_true = y_true.as_matrix()
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
    print('HammingLoss*:         %.2f' % hammingLoss);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print("-" * 10);


if __name__ == '__main__':

    # ## Part 1 ************** each time using one user to predit the test data *****************
    # indir = 'data_csv/train/10users'
    # for root, dirs, filenames in os.walk(indir):
    #     for f in filenames:
    #         train = pd.read_csv(os.path.join(root, f))
    #         test = pd.read_csv('data_csv/train/33A85C34-CFE4-4732-9E73-0A7AC861B27A.features_labels.csv')
    #
    #         (train_features, train_labels)=features_labels_process(train)
    #         (test_features, test_labels) = features_labels_process(test)
    #
    #         print (train_features.shape, train_labels.shape)
    #         print (test_features.shape, test_labels.shape)
    #
    #         ## ------- compare cosine similarity between train and test labels
    #
    #         # similarity  = cosine_similarity([train_labels, test_labels])
    #         # print (similarity)
    #
    #         ## ------- Fit an independent logistic regression model or MLP for each class using the OneVsRestClassifier wrapper.
    #         # ovr = OneVsRestClassifier(LogisticRegression())
    #         ovr = OneVsRestClassifier(MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,learning_rate_init=.1))
    #         # ------ evaluation of test data predicion  -----------------------#
    #
    #         ovr.fit(train_features, train_labels)
    #         Y_pred = ovr.predict(test_features)

            # evaluation(Y_pred, test_labels)


    ## Part 2 ************** fixed train & test prediction *****************

    train = pd.read_csv('data_csv/train/train1.csv')
    test = pd.read_csv('data_csv/train/33A85C34-CFE4-4732-9E73-0A7AC861B27A.features_labels.csv')

    (train_features, train_labels) = features_labels_process(train)
    (test_features, test_labels) = features_labels_process(test)

    print (train_features.shape, train_labels.shape)
    print (test_features.shape, test_labels.shape)

    ## ------- validation -- split the train set
    # X_train, X_test, Y_train, Y_test = train_test_split(train_features, train_labels, test_size=.9,random_state=0)

    # # ------ evaluation of five-folder cross validation -----------------------#
    # ovr.fit(X_train, Y_train)
    # Y_valid = ovr.predict(X_test)
    #
    # evaluation(Y_valid, Y_test)


    ## ------- Fit an independent logistic regression model or MLP for each class using the OneVsRestClassifier wrapper.

    # ovr = OneVsRestClassifier(svm.SVC())
    # ovr = OneVsRestClassifier(LogisticRegression())
    ovr = OneVsRestClassifier(
        MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
                      max_iter=10, learning_rate_init=.1))
    # ovr = OneVsRestClassifier(MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50,50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1))

    # ------ evaluation of test data predicion  -----------------------#

    ovr.fit(train_features, train_labels)
    Y_pred = ovr.predict(test_features)

    evaluation(Y_pred, test_labels)




