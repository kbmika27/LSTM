#coding:utf-8
#ファイルへの書き込み

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import SGD

#データセットを作る　入力は3*n,出力は3*n
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
    #ここから
    trainX,trainY=[],[]
    filenum=glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/x_data/*")  #ファイル数を取得する
    filenum=len(filenum)
    for i in range(filenum-1): #200-13
        number = i
        filename = "x_data/x_%d.txt" % number
        with open(filename, mode='w') as file:  #書き込み
            for j in range(13):
                if (j<12):
                   file.write(str(dataset[i+j][0])+'\n')
                else:
                   file.write(str(dataset[i + j][0]) )
    file.close()
    for i in range(filenum):
        a=[]
        text="x_data/x_%d.txt" % (i)
        f = open(text)  # ファイルの読み込み
        alldata = f.read()  # 全部読み込む
        scaler = MinMaxScaler(feature_range=(0, 1))
        f.close()
        lines1 = alldata.split('\n')  # 改行で区切る
        for line in lines1:
            data2 = line
            data2 = data2.replace('"', '')
            a.append([float(data2)])  # aはtrainとtestにあたる
        a=scaler.fit_transform(a)  #これを正規化するので合ってるのか？
        a=a.tolist()
        print("a"+str(a))
        train_in,train_out = train_test_split(np.array(a), test_size=0.2, shuffle=False)  # 10と3 入力と出力
        trainX.append(train_in)
        trainY.append(train_out)
    print("trainX: " + str(trainY))

    # encoder,decoder
    hidden_size = 4  # 隠れ層
    encoder = Encoder(1, hidden_size, 1)
    decoder = Decoder(1, hidden_size, 3)
    criterion = nn.MSELoss()
    encoder_optimizer = SGD(encoder.parameters(), lr=0.01)  # optimizerの初期化
    decoder_optimizer = SGD(decoder.parameters(), lr=0.01)
    # 学習開始
    batch_size = 10
    for epoch in range(100):
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(len(trainX) / batch_size)):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # d = torch.tensor([trainX[i]]).float()  # 入力
            # label = torch.tensor([trainY[i]]).float() #出力 1*3*1
            d, label = create_batch(trainX, trainY, batch_size)  # dは3次元
            encoder_output, encoder_hidden = encoder(d)  # エンコーダーを通過
            decoder_hidden = encoder_hidden  # 1*1*4
            decoder_output, decoder_hidden = decoder(label, decoder_hidden)
            # print("decoder"+str(decoder_output))
            # print(type(decoder_output))
            label_output = []
            for i in range(batch_size):  # (5*3*1)を型変換
                a = []
                for j in range(3):
                    a.append(label[i][j].data.item())
                label_output.append(a)
            label_output = np.array(label_output)
            label_output = torch.tensor([label_output]).float()
            loss = criterion(decoder_output, label_output)  # 損失計算
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        if (epoch % 10 == 0):
            print("loss: " + str(running_loss))

    #ここまで
    # 正規化 とりあえず入れてみたら動いたけど本当に合ってるのか確認


if __name__ == '__main__':
    main()