"""
Tensorflow LSTM

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

#define parameters
rnn_unit=10       #hidden layer units
input_size=9
output_size=6
lr=0.006         #learning rate

#—————————————————— get dataset ——————————————————————
f=open('data/preprocessed_data/bj_merged.csv')
df=pd.read_csv(f)     #读入股票数据
df.set_index('utc_time', inplace=True)
dataframe = df.fillna(0)
data = dataframe.values
# integer encode direction
encoder = LabelEncoder()
data[:, 8] = encoder.fit_transform(data[:, 8])
data[:, 7] = encoder.fit_transform(data[:, 7])


#获取训练集
def get_train_data(batch_size,time_step,train_begin,train_end):
    batch_index=[]
    data_train=data[train_begin:train_end]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(data_train)
    # normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(scaled_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=scaled_train_data[i:i+time_step,:-6]
       y=scaled_train_data[i:i+time_step,-6:]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(scaled_train_data)-time_step))
    # print (train_y[0:10], train_x[0:10])

    return batch_index,train_x,train_y


#获取测试集
def get_test_data(time_step,test_begin,test_end):
    data_test=data[test_begin:test_end]

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_test_x = scaler_x.fit_transform(data_test[:,:-6])
    scaled_test_y = scaler_y.fit_transform(data_test[:,-6:])
    print (scaled_test_y.shape)
    print (scaled_test_x.shape)

    size=(len(scaled_test_x)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=scaled_test_x[i*time_step:(i+1)*time_step,:]
       y=scaled_test_y[i*time_step:(i+1)*time_step]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((scaled_test_x[(i+1)*time_step:,:]).tolist())
    test_y.extend((scaled_test_y[(i+1)*time_step:]).tolist())
    print (len(test_y))

    return scaler_y,test_x,test_y

def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))

def plot(y_true, y_predict):

    columns_predict = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    for i in range (len(columns_predict)):
        plt.figure(figsize=(24, 8))
        plt.plot(y_true[:,i], c='g', label='Actual')
        plt.plot(y_predict[:,i], c='r',  label='Predicted')
        plt.legend(fontsize='small')
        plt.title('Actual and Predicted ' + columns_predict[i])
        plt.savefig('Results/predicted_and_actural_'+columns_predict[i]+'.eps', format="eps", dpi=1000)
        plt.show()


#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,6]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[6,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.LSTMCell(rnn_unit,state_is_tuple=True)
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#——————————————————训练模型——————————————————
def train_lstm(batch_size,time_step,train_begin,train_end):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)

    #loss function
    print (pred.shape,Y.shape)
    print (type(Y))
    print (type(pred))

    # loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    loss = tf.losses.mean_squared_error(tf.reshape(pred,[-1]),tf.reshape(Y, [-1]))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)


    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #重复训练10000次
        epoch=100
        last_loss = 0
        for i in range(epoch):
            for step in range(len(batch_index)-1):
                _,loss_ = sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],
                                                            Y:train_y[batch_index[step]:batch_index[step+1]]})
            print("Number of iterations:", i, " loss:", loss_)

            # early stopping
            patience = 20
            min_delta = 0.001
            if epoch>0 and loss_ - last_loss > min_delta:
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt > patience:
                print("early stopping...")
                break
            last_loss = loss_
        print("model_save: ", saver.save(sess, 'model_save2/modle.ckpt'))


#————————————————预测模型————————————————————
def prediction(time_step,test_begin,test_end):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    scaler_y,test_x,test_y=get_test_data(time_step,test_begin,test_end)

    with tf.variable_scope("sec_lstm", reuse=True):
        pred,_=lstm(X)

    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)

        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          test_predict.extend(prob)

        test_predict = scaler_y.inverse_transform(test_predict)
        test_y = scaler_y.inverse_transform(test_y)

        if len(test_predict) < len(test_y):
            test_y = test_y[:len(test_y) - (len(test_y)-len(test_predict))]

        rmse = np.sqrt(mean_squared_error(test_predict, test_y))
        mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
        print('mae:', mae, 'rmse:', rmse)

        Smape = smape(test_y, test_predict)
        print('Test Smape: %.3f' % Smape)

        plot(test_y, test_predict)

if __name__ == '__main__':

    train_lstm(batch_size=72, time_step=24, train_begin=0, train_end=70000)
    prediction(time_step=24,test_begin=70000, test_end=73000)