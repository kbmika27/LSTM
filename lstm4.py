#coding:utf-8
#attentionなしのLSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras import Sequential
from keras.layers import LSTM, Dense
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#x,y,tをエクセルから読み込み、それぞれ配列に入れる
x=pd.read_excel('lstm_data.xlsx',usecols=[1]) #skiprows=で上を読まない
datax=x.values
x=[]
for i in range(len(datax)):
    x.append(datax[i][0])
print(x)
y=pd.read_excel('lstm_data.xlsx',usecols=[2]) #skiprows=で上を読まない
datay=y.values
y=[]
for i in range(len(datay)):
    y.append(datay[i][0])
print(y)
t=pd.read_excel('lstm_data.xlsx',usecols=[0])
time=t.values
t=[]
for i in range(len(time)):
    t.append(time[i][0])
#3Dでplot
fig=plt.figure()
ax=Axes3D(fig)
ax.plot(x,t,y)
print("型")
print(type(x))
ax.set_xlabel("x")
ax.set_ylabel("time")
ax.set_zlabel("y")
#plt.show()#表示
dataset=np.array([[0]*2]*len(x))#x,yをセットにしたデータセット
for i in range(len(x)):
    dataset[i][0]=x[i]
    dataset[i][1]=y[i]
np.random.seed(7)
#正規化
#dataset=(dataset-datamin).astype(float)/(dataset-datamax).astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset=scaler.fit_transform(dataset)
print(dataset)
#トレーニングデータとテスト用データに分ける
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
#データセットを作る
def create_dataset(dataset,maxlen):
    #dataX = np.array([[[1.0] * 2] * maxlen] * (len(dataset)-maxlen-1)) #1つの番地に3つの値を持つ
    dataX=np.array([[[1.0 for i in range(2)]for j in range (maxlen)]for k in range(len(dataset)-maxlen-1)])
    print(dataX)
    dataY = np.array([[1.0] * 2] * (len(dataset)-maxlen-1))
    print("青")
    for i in range(len(dataset) - maxlen - 1):
        #a = np.array([[0] * 2] * maxlen)  # dataXの中の1つのセット aもdataX[i]も長さ3
        for j in range(maxlen):
           # a[j][0]=dataset[j][0]
           # a[j][1]=dataset[j][1]  aはいらないかも
           dataX[i][j][0]=dataset[i+j][0]
           dataX[i][j][1]=dataset[i+j][1]
        print("い")
        print(dataX[i][j])
        dataY[i]=dataset[i+maxlen]
    print("う")
    print(dataX)
    return np.array(dataX), np.array(dataY)
#reshape ,X=t and Y=t+maxlen
maxlen=3 #t,t-1,t-2のデータセットを作成
trainX,trainY = create_dataset(train, maxlen) #trainX=10,3,2
testX, testY = create_dataset(test, maxlen)
print("お")
print(trainX[:10,:])
#print(trainY[:10])#データの冒頭の確認

#入力の変形
trainX = np.reshape(trainX, (trainX.shape[0], 2, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 2, testX.shape[1]))

#lstmの学習モデルを作成する  ここにattentionを入れる
model = Sequential()
model.add(LSTM(4, input_shape=(2, maxlen)))#4は隠れ層  input_shape=1,maxlenだったところを変更
model.add(Dense(2))#ニューロンの数を調節している
model.compile(loss='mean_squared_error', optimizer='adam')#誤差関数、最適化法
#学習。lstmを頭良くさせた
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2) #epocが増えるとlossが減る batchでtrainingdataを分割
#学習データで予測 評価
trainPredict = model.predict(trainX)
trainPredict=scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
#未来の予測
testPredict=model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
print("二乗")
print(trainPredict)
print(testY)
#二乗平均平方根誤差を計算
tra_x=[] #trainpredictのx
tra_y=[]
train_x=[] #trainyのx
train_y=[]
for i in range(len(trainPredict)):  #trainPredictとtrainY
    tra_x.append(trainPredict[i][0])
    train_x.append(trainY[i][0])
    tra_y.append(trainPredict[i][1])
    train_y.append(trainY[i][1])
trainscore_x=mean_squared_error(tra_x,train_x)  #trainのxの2乗
trainscore_y=mean_squared_error(tra_y,train_y)  #trainのyの2乗
trainscore=math.sqrt(trainscore_x+trainscore_y)
print('Train Score: %.2f RMSE' % (trainscore))
tes_x=[]
tes_y=[]
test_x=[]
test_y=[]
for i in range(len(testPredict)):
    tes_x.append(testPredict[i][0])
    test_x.append(testY[i][0])
    tes_y.append(testPredict[i][1])
    test_y.append(testY[i][1])
testscore_x=mean_squared_error(tes_x,test_x)
testscore_y=mean_squared_error(tes_y,test_y)
testscore=math.sqrt(testscore_x+testscore_y)
print('Test Score: %.2f RMSE' % (testscore))
#シフトトレイン予測
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :]=np.array([[0]*2]*len(x),dtype=float)#NANの生成
trainPredictPlot[maxlen:len(trainPredict)+maxlen, :] =trainPredict #trainpredictがfloat64
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.array([[0]*2]*len(x),dtype=float)
testPredictPlot[len(trainPredict)+(maxlen*2)+1:len(dataset)-1, :] = testPredict
print(testPredictPlot)
ax_x=[]
ax_y=[]
ax_t=[]
for i in range(maxlen,len(trainPredict)+maxlen,1):
    ax_x.append(trainPredictPlot[i][0])
    ax_y.append(trainPredictPlot[i][1])
    ax_t.append(i+1)
ax.plot(ax_x,ax_t,ax_y)
ax_tx=[]
ax_ty=[]
ax_tt=[]
for i in range(len(trainPredict)+(maxlen*2)+1,len(dataset)-1,1):
    ax_tx.append(testPredictPlot[i][0])
    ax_ty.append(testPredictPlot[i][1])
    ax_tt.append(i+1)
ax.plot(ax_tx,ax_tt,ax_ty)
plt.show()

