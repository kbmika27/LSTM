#coding:utf-8
#pytorchの1次元version
from Tkinter import Variable

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.python.estimator import keras
from torch.optim import SGD

#データセットを作る
def create_dataset(dataset,maxlen):
    dataX,dataY=[],[]
    for i in range(len(dataset) - maxlen - 1):
        a = dataset[i:(i + maxlen)]
        dataX.append(a)
        dataY.append(dataset[i + maxlen+1])
    print("datax")
    print(dataX)
    print("datay")
    print(dataY)
    return np.array(dataX), np.array(dataY)

#モデル定義
class Predictor(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Predictor,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
        self.output_layer=nn.Linear(hiddenDim,outputDim)  #Denseのようなもの

    def forward(self, inputs, hidden0=None): #予測をする
        output,(hidden, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル
        output = self.output_layer(output[:, -1, :])
        return output

#バッチ化する
def create_batch(trainx, trainy, batch_size=10):
    #trainX、trainYを受け取ってbatchX、batchYを返す
    batchX=[]
    batchY=[]
    for _ in range(batch_size):
        idx=np.random.randint(0,len(trainx)-1)
        batchX.append(trainx[idx])
        batchY.append(trainy[idx])
    return torch.tensor(batchX).float(),torch.tensor(batchY).float()

#mainメソッド
def main():
    dataframe = pd.read_excel('x_data.xlsx', usecols=[1])  # skiprows=で上を読まない
    dataset = dataframe.values
    dataset = dataset.astype("float32")
    plt.plot(dataset)
    np.random.seed(7)
    # 正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)  # この段階でdatasetはnumpy.ndarray
    dataset = dataset.tolist()
    print(dataset)
    # トレーニングデータとテスト用データに分ける
    train_size = int(len(dataset) * 0.7)  # 14
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    print(len(train), len(test))  # 35,15
    print(train)
    print("testdata")
    print(test)

    # reshape ,X=t and Y=t+maxlen
    maxlen = 3
    trainX, trainY = create_dataset(train, maxlen)  # データセットの作成
    testX, testY = create_dataset(test, maxlen)
    print(trainX[:10, :])
    print(trainY[:10])  # ここの書き方？
    print("あ")
    print(type(trainX[0]))
    # 入力の変形
    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # print("trainX")
    print(len(trainX))
    hidden_size = 4  # 隠れ層
    # batch_size = 1 #いくつに分けるか
    model = Predictor(1, hidden_size, 1)  # modelの宣言
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)  # model.compile
    # 学習開始
    test_accuracy = 0.0
    train_accuracy = 0.0
    batch_size = 5
    for epoch in range(400):  # training model.fit
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(len(trainX)/batch_size)):  # 10
            optimizer.zero_grad()
            #d = torch.tensor([trainX[i]]).float()  # ここ変えた d.ndimension=3
            #label = torch.tensor([trainY[i]]).float()
            d,label=create_batch(trainX,trainY,batch_size) #dは3次元
            output = model(d)  # tensor([[0.4513]]) だったのがoutput:tensor([[0.4040]], grad_fn=<AddmmBackward>)
            #print("d:" + str(output.size()))
            #print("label:" + str(label.size()))
            loss = criterion(output, label)  # 損失計算　labelは多分元のデータ
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        training_accuracy /= len(trainX)
        if(epoch%50==0):
            print('%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))

    # training 評価
    trainPredict = []  # ここにlabelを入れていく
    for i in range(int(len(trainX) / batch_size)):
        offset = i * batch_size
        traindata = torch.tensor(trainX[offset:offset + batch_size]).float()
        trainlabel = torch.tensor(trainY[offset:offset + batch_size]).float()
        trainoutput = model(traindata)  # 学習したもの
        for i in range(len(trainoutput)):
            train_value = trainoutput[i].data.item() # 値を取得
            trainPredict.append(train_value) #値をlistに追加
        train_accuracy += np.sum(np.abs((trainoutput.data - trainlabel.data).numpy()) )
    training_accuracy /= len(trainX)
    print("評価trainiaccuracy" + str(training_accuracy))


    # test　評価
    testPredict = []  # ここにlaelを入れていく
    for i in range(int(len(testX) /batch_size)):
        offset = i * batch_size
        testdata = torch.tensor(testX[offset:offset + batch_size]).float()
        testlabel = torch.tensor(testY[offset:offset + batch_size]).float()
        testoutput = model(testdata)
        for i in range(len(testoutput)):
            test_value = testoutput[i].data.item()  # 値を取得
            testPredict.append(test_value) #値をlistに追加
        test_accuracy += np.sum(np.abs((testoutput.data - testlabel.data).numpy()) )
    test_accuracy/=len(testX)
    print("評価test_accuracy: " + str(test_accuracy))

    # trainのplot
    trainPredict = np.array(scaler.inverse_transform([trainPredict]))
    trainPredict = np.reshape(trainPredict, (len(trainPredict[0]), 1))
    #print("trainpredict" + str(trainPredict))
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan  # NANの生成
    trainPredictPlot[maxlen:len(trainPredict) + maxlen, :] = trainPredict
    # print("trainplot"+str(trainPredictPlot))

    # testのplot
    test_accuracy /= len(testX)
    testPredict = np.array(scaler.inverse_transform([testPredict]))  # 正規化から元に戻す
    testPredict = np.reshape(testPredict, (len(testPredict[0]), 1))  # 10*1の形に変形
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan  # NANの作成
    testPredictPlot[len(trainPredict) + (maxlen * 2) + 1:len(trainPredict) + (maxlen * 2) + 1+len(testPredict), ] = testPredict #testpredictは10

    # loss等
    print(training_accuracy)
    print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy, test_accuracy))
    plt.plot(scaler.inverse_transform(dataset), color="g", label="row")
    plt.plot(trainPredictPlot, color="b", label="trainpredict")  # 全部データタイプはnp.ndarrayだった
    plt.plot(testPredictPlot, color="m", label="testpredict")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()