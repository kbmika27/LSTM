#coding:utf-8
#1次元のLSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Reshape, Embedding, Flatten, GlobalMaxPooling1D, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.python.estimator import keras

dataframe = pd.read_excel('x_data.xlsx',usecols=[1]) #skiprows=で上を読まない
dataset=dataframe.values
dataset=dataset.astype("float32")
plt.plot(dataset)
np.random.seed(7)
#正規化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset=scaler.fit_transform(dataset)
print(dataset)
#トレーニングデータとテスト用データに分ける
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))#14,6
#データセットを作る
def create_dataset(dataset,maxlen):
    dataX,dataY=[],[]
    for i in range(len(dataset) - maxlen-1):
        a = dataset[i:(i + maxlen), 0]
        dataX.append(a)
        dataY.append(dataset[i + maxlen+1, 0]) #ここi+maxlenだったところを変更した
    print("datax")
    print(dataX)
    print("datay")
    print(dataY)
    return np.array(dataX), np.array(dataY)

#reshape ,X=t and Y=t+maxlen
maxlen=3
trainX,trainY = create_dataset(train, maxlen)
testX, testY = create_dataset(test, maxlen)
print(trainX[:10,:])
print(trainY[:10])#ここの書き方？
print("あ")
print(trainX.shape[0])
#入力の変形
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("trainX"+str(len(trainX)))
print(trainX)


#lstmの学習モデルを作成する
model = Sequential()
model.add(LSTM(4, input_shape=(1, maxlen)))#4は隠れ層 2次元
#model.add(Embedding(100,64))
#model.add(SeqSelfAttention(attention_activation='softmax')) #3次元じゃないとだめ
model.add(Dense(1,name='dense'))#ニューロンの数を調節している 全結合ネットワーク
model.compile(loss='mean_squared_error', optimizer='adam')#誤差関数、最適化法
model.summary()
#学習。lstmを頭良くさせた
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2) # batchでtrainingdataを分割 verboは1行ごと表示
#学習データで予測 評価
trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
print("trainPredict"+str(len(trainX)))
print(trainPredict)
trainY = scaler.inverse_transform([trainY])
print("trainY")
print(trainY) #16-3-1個元に戻した値が入ってた
#未来の予測
testPredict=model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
#二乗平均平方根誤差を計算
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
#シフトトレイン予測
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan #NANの生成
trainPredictPlot[maxlen:len(trainPredict)+maxlen, :] = trainPredict
print(trainPredictPlot)
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(maxlen*2)+1:len(dataset)-1, :] = testPredict
print(testPredictPlot)
plt.plot(scaler.inverse_transform(dataset), color ="g", label = "row")
plt.plot(trainPredictPlot,color="b", label="trainpredict")
plt.plot(testPredictPlot,color="m",label="testpredict")

plt.legend()
plt.show()