#coding:utf-8
#pytorchでattentionを組み込む　1次元ver encoder decoder

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import SGD
#データセットを作る　入力は10*n,出力は3*n
def create_dataset(dataset,len_x,len_y):
    dataX,dataY=[],[] #入力と出力
    for i in range(len(dataset) - len_x - len_y):
        a = dataset[i:(i + len_x)]
        dataX.append(a)
    for i in range(len(dataset)-len_x-len_y):
        b = dataset[i+len_x+1:(i+len_x+len_y+1)]
        dataY.append(b)
    print("datax")
    print(dataX)
    print("datay")
    print(dataY)
    return np.array(dataX), np.array(dataY)
#エンコーダー
class Encoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Encoder,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)

    def forward(self, inputs, hidden0=None): #予測をする
        output,(state, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル
        return output,(state,cell)

#デコーダー
class Decoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Decoder,self).__init__()
        self.hiddenDim=hiddenDim
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)  # 全結合層

    def forward(self, inputs, hidden0): #予測をする
        output,(hidden,cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル
        output = self.output_layer(output[:, -1, :])
        return output,hidden

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
    train, test = train_test_split(dataset, test_size=0.3,shuffle=False)
    print(train)
    # データセットの作成
    len_x = 10
    len_y = 3
    trainX, trainY = create_dataset(train, len_x, len_y)  #入力が10で出力が3  Yは127*3*1
    # testX, testY = create_dataset(test, len_x ,len_y)
    print("trainY: " + str(trainY[:10, :]))
    print(len(trainX), len(trainY))

    # encoder,decoder
    hidden_size = 4  # 隠れ層
    encoder = Encoder(1, hidden_size, 1)
    decoder = Decoder(1, hidden_size, len_y) #len_yは出力される長さ
    criterion = nn.MSELoss()
    encoder_optimizer = SGD(encoder.parameters(), lr=0.01)  # optimizerの初期化
    decoder_optimizer = SGD(decoder.parameters(), lr=0.01)

    # 学習開始
    batch_size = 10
    for epoch in range(50):
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(len(trainX) / batch_size)):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            #d = torch.tensor([trainX[i]]).float()  # 入力 1*10*1
            #label = torch.tensor([trainY[i]]).float() #出力 1*3*1
            d, label = create_batch(trainX, trainY, batch_size)  # dは3次元
            encoder_output, encoder_hidden = encoder(d)  # エンコーダーを通過
            decoder_hidden = encoder_hidden  # 1*1*4 tensor([[[ 0.0478,  0.1327,  0.1693, -0.4749]]]がたくさん
            decoder_output, decoder_hidden = decoder(label, decoder_hidden)
            print("decoder"+str(decoder_output))
            print("label"+str(label))
            label_output = []
            for i in range(batch_size):  # (5*3*1)を型変換
                a = []
                for j in range(len_y):
                    a.append(label[i][j].data.item())
                label_output.append(a)
            label_output = np.array(label_output)
            label_output = torch.tensor([label_output]).float()
            loss = criterion(decoder_output, label_output)  # 損失計算
            print("decoder_output"+str(decoder_output))
            print("lavel"+str(label_output))
            print("ロス"+str(loss))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        if (epoch % 10 == 0):
            print("loss: " + str(running_loss))


if __name__ == '__main__':
    main()