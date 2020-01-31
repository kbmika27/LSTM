#coding:utf-8
#sequence to sequence 学習用
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.model_selection import train_test_split
#エンコーダー
class Encoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Encoder,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
    def forward(self, inputs, hidden0=None): #予測をする
        output,(state, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル
        return output,state,cell
#デコーダー
class Decoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Decoder,self).__init__()
        self.hiddenDim=hiddenDim
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)  # 全結合層
    def forward(self, inputs, hidden0): #予測をする
        output,(hidden,cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル ここには全部コピーを入れる
        att_output=output # コピーしておく
        output = self.output_layer(output[:, -1, :])
        return output,hidden,att_output
#バッチ化する
def create_batch(trainx, trainy, batch_size=10):
    batchX=[]
    batchY=[]
    for _ in range(batch_size):
        idx=np.random.randint(0,len(trainx)-1)
        batchX.append(trainx[idx])
        batchY.append(trainy[idx])
    return torch.tensor(batchX).float(),torch.tensor(batchY).float()
def main():
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    numline=sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt')) #13
    trainX=[]
    trainY=[]
    for i in range(int(filenum*0.8)): #ファイルの読み込み
        j=0 #カウント用
        a=np.array([[1.0] * 2] * numline)
        text="xy6_data/xy_%d.txt" % (i)
        f = open(text)  # ファイルを開く
        alldata = f.read()  # xy_i.txtを全部読み込む
        scaler = MinMaxScaler(feature_range=(0, 1)) #正規化の準備
        f.close()
        lines = alldata.split('\n')  # 改行で区切る
        for line in lines: #1行ずつ読み込み
            linedata =line.split(',')
            line_x=linedata[0] #各行のx座標
            line_y=linedata[1] #各行のy座標
            a[j][0]=float(line_x)
            a[j][1]=float(line_y)
            j+=1
        a = scaler.fit_transform(a)
        a = a.tolist()
        train_in, train_out = train_test_split(np.array(a), test_size=0.5, shuffle=False)
        trainX.append(train_in)
        trainY.append(train_out)
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    hidden_size = 4  # 隠れ層
    encoder = Encoder(2, hidden_size,2)
    decoder = Decoder(2, hidden_size, 20)
    criterion = nn.MSELoss()
    encoder_optimizer = SGD(encoder.parameters(), lr=0.01)  # optimizerの初期化
    decoder_optimizer = SGD(decoder.parameters(), lr=0.01)
    # 学習開始
    batch_size = 15
    for epoch in range(150):
        running_loss = 0.0
        for i in range(int(len(trainX) / batch_size)):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            #d = torch.tensor([trainX[i]]).float()  # 入力 1*10*2 バッチ化しない時利用
            #label = torch.tensor([trainY[i]]).float()
            d,label=create_batch(trainX,trainY,batch_size)
            encoder_output,encoder_state,encoder_cell=encoder(d) #encoder通過
            #ここでencoder_outputを渡したい
            encoder_hidden=(encoder_state,encoder_cell)
            decoder_hidden=encoder_hidden
            decoder_output, decoder_hidden,decoder_attention = decoder(label, decoder_hidden) #decoder通過
            label_output=[]
            for k in range(len(label)):
                a=[]
                for j in range(len(train_out)):
                    a.append(label[k][j][0].data.item())
                    a.append(label[k][j][1].data.item())
                label_output.append(a)
            label_output = torch.tensor(label_output).float()
            loss=criterion(decoder_output,label_output)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        if (epoch % 10 == 0):
            print("loss: " + str(running_loss))
    #モデルの保存
    torch.save(encoder.state_dict(), 'modelstore/seq2_en_model')
    torch.save(decoder.state_dict(), 'modelstore/seq2_de_model')
    print("学習終了")
if __name__ == '__main__':
    main()