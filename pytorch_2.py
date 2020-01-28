#coding:utf-8
#pytorchの2次元LSTM attentionなし LSTM実験用
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import SGD

#データセットを作る
def create_dataset(dataset,maxlen):
    dataX = np.array([[[1.0 for i in range(2)] for j in range(maxlen)] for k in range(len(dataset) - maxlen - 1)]) #全てに1.0を一旦入れる
    dataY = np.array([[1.0] * 2] * (len(dataset) - maxlen - 1))  #dataXを入れると出力されるもの
    #データを入れるfor文
    for i in range(len(dataset) - maxlen - 1): #105-6-1
        #a = np.array([[0] * 2] * maxlen)  # dataXの中の1つのセット aもdataX[i]も長さ3
        for j in range(maxlen):
           # a[j][0]=dataset[j][0]
           # a[j][1]=dataset[j][1]  aはいらないかも
           dataX[i][j][0]=dataset[i+j][0]
           dataX[i][j][1]=dataset[i+j][1]
        dataY[i]=dataset[i+maxlen]
    return np.array(dataX), np.array(dataY)

#モデル定義
class Predictor(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Predictor,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
        self.output_layer=nn.Linear(hiddenDim,outputDim)  #Denseのようなもの

    def forward(self, inputs, hidden0=None): #予測をする
        output,(hidden, cell) = self.rnn(inputs, hidden0)  # LSTM層
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

def main():
    # x,y,tをエクセルから読み込み、それぞれ配列に入れる
    x = pd.read_excel('lstm_data.xlsx', usecols=[1])  # skiprows=で上を読まない
    datax = x.values
    x = []
    for i in range(len(datax)):
        x.append(datax[i][0])
    y = pd.read_excel('lstm_data.xlsx', usecols=[2])  # skiprows=で上を読まない
    datay = y.values
    y = []
    for i in range(len(datay)):
        y.append(datay[i][0])
    print("y" + str(y))
    t = pd.read_excel('lstm_data.xlsx', usecols=[0])
    time = t.values
    t = []
    for i in range(len(time)):
        t.append(time[i][0])
    # 3Dでplot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, t, y)
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_zlabel("y")

    dataset = np.array([[0] * 2] * len(x))  # x,yをセットにした2次元のデータセット
    for i in range(len(x)):
        dataset[i][0] = x[i]
        dataset[i][1] = y[i]
    np.random.seed(7)
    #正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # トレーニングデータとテスト用データに分ける
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset), :]

    # reshape ,X=t and Y=t+maxlen
    maxlen=6 #t,t-1,t-2のデータセットを作成
    trainX, trainY = create_dataset(train, maxlen)  # trainX=98*maxlen*2 trainY=98*2
    print("trainX"+str(trainY.shape))
    testX, testY = create_dataset(test, maxlen)
    print("trainX"+str(trainX[:10, :]))
    #trainX = np.reshape(trainX, (trainX.shape[0], 2, trainX.shape[1])) #入力の変形 これ入れるとエラーになる
    #testX = np.reshape(testX, (testX.shape[0], 2, testX.shape[1]))

    #modelの宣言
    hidden_size = 4  # 隠れ層
    model = Predictor(2, hidden_size, 2)  # modelの宣言
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)  # model.compile

    #学習開始
    test_accuracy = 0.0
    train_accuracy = 0.0
    batch_size = 5
    for epoch in range(150):  # training model.fit
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(len(trainX)/batch_size)):  # 10
            optimizer.zero_grad()
            #d= torch.tensor([trainX[i]]).float()  # ここ変えた d.ndimension=3
            #label = torch.tensor([trainY[i]]).float()
            d, label = create_batch(trainX, trainY, batch_size)  # dは3次元
            #print("label: "+str(label))
            output = model(d)  # tensor([[0.4513]]) だったのがoutput:tensor([[0.4040]], grad_fn=<AddmmBackward>)
            loss = criterion(output, label)  # 損失計算　labelは多分元のデータ
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        training_accuracy /= len(trainX)
        if (epoch %20 == 0):
            print('%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))
    print("学習終了")

    # training
    trainPredict_x= []  #ここにlabelのxを入れる
    trainPredict_y= []  #ここにlabelのyを入れる
    for i in range(int(len(trainX) / batch_size)):
        offset = i * batch_size
        traindata = torch.tensor(trainX[offset:offset + batch_size]).float()
        trainlabel = torch.tensor(trainY[offset:offset + batch_size]).float()
        trainoutput = model(traindata)  # 学習したもの
        #print("trainoutput: "+str(trainoutput))
        for i in range(len(trainoutput)):
            train_value_x= trainoutput[i][0].data.item() # 値を取得
            train_value_y= trainoutput[i][1].data.item()
            trainPredict_x.append(train_value_x)  #値をlistに追加
            trainPredict_y.append(train_value_y)
        train_accuracy += np.sum(np.abs((trainoutput.data - trainlabel.data).numpy()) < 0.1)
    training_accuracy /= len(trainX)
    print("評価trainiaccuracy" + str(training_accuracy))
    trainPredict = np.array([[1.0] * 2] * len(trainPredict_x))  # 2次元のtrainPredict
    for i in range(len(trainPredict_x)): #2次元配列にする
        trainPredict[i][0]=trainPredict_x[i]
        trainPredict[i][1]=trainPredict_y[i]
    trainPredict = np.array(scaler.inverse_transform(trainPredict)) #正規化を元に戻す
    for i in range(len(trainPredict_x)):
        trainPredict_x[i]=trainPredict[i][0]
        trainPredict_y[i]=trainPredict[i][1]
    #trainのplot
    trainPredict_t=[] #対応する時間を入れる
    for i in range(maxlen, len(trainPredict) + maxlen, 1):
        trainPredict_t.append(i+1)
    ax.plot(trainPredict_x,trainPredict_t,trainPredict_y)

    #test
    testPredict_x = []  # ここにlabelのxを入れる
    testPredict_y = []  # ここにlabelのyを入れる
    for i in range(int(len(testX) / batch_size)):
        offset = i * batch_size
        testdata = torch.tensor(testX[offset:offset + batch_size]).float()
        testlabel = torch.tensor(trainY[offset:offset + batch_size]).float()
        testoutput = model(testdata)  # modelに入れて返ってくる値
        for i in range(len(testoutput)):
            test_value_x= testoutput[i][0].data.item() # 値を取得
            test_value_y= testoutput[i][1].data.item()
            testPredict_x.append(test_value_x)  #値をlistに追加
            testPredict_y.append(test_value_y)
        test_accuracy += np.sum(np.abs((testoutput.data - testlabel.data).numpy()) )
    test_accuracy /= len(testX)
    print("評価test_accuracy: " + str(test_accuracy))
    testPredict = np.array([[1.0] * 2] * len(testPredict_x))  # 2次元のtestPredict
    for i in range(len(testPredict_x)): #2次元配列にする
        testPredict[i][0]=testPredict_x[i]
        testPredict[i][1]=testPredict_y[i]
    testPredict = np.array(scaler.inverse_transform(testPredict)) #正規化を元に戻す
    print(testPredict)
    print(testPredict.shape)
    for i in range(len(testPredict_x)):
        testPredict_x[i]=testPredict[i][0]
        testPredict_y[i]=testPredict[i][1]

    #testのplot
    testPredict_t = []  # 対応する時間を入れる
    for i in range(len(trainPredict) + (maxlen * 2) + 1, len(trainPredict) + (maxlen * 2) + 1+len(testPredict), 1):
        testPredict_t.append(i+1)
    ax.plot(testPredict_x,testPredict_t,testPredict_y)



    plt.show()  # 表示
if __name__ == "__main__":
    main()
