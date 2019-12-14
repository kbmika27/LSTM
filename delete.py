#coding:utf-8
#pytorchの2次元ver encoder decoder
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
    x = pd.read_excel('xy_data.xlsx', usecols=[1])  # skiprows=で上を読まない
    datax = x.values
    x = []
    for i in range(len(datax)):
        x.append(datax[i][0])
    y = pd.read_excel('xy_data.xlsx', usecols=[2])  # skiprows=で上を読まない
    datay = y.values
    y = []
    for i in range(len(datay)):
        y.append(datay[i][0])
    dataset = np.array([[0] * 2] * len(x))  # x,yをセットにした2次元のデータセット
    for i in range(len(x)):
        dataset[i][0] = x[i]
        dataset[i][1] = y[i]
    np.random.seed(7)
    #ここから
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    numline=sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy_data/xy_0.txt')) #13
    trainX=[]
    trainY=[]
    for i in range(filenum): #ファイルの読み込み
        j=0 #カウント用
        a=np.array([[1.0] * 2] * numline)
        text="xy_data/xy_%d.txt" % (i)
        f = open(text)  # ファイルを開く
        alldata = f.read()  # xy_i.txtを全部読み込む
        scaler = MinMaxScaler(feature_range=(0, 1)) #正規化の準備
        f.close()
        lines = alldata.split('\n')  # 改行で区切る
        for line in lines: #1行
            linedata =line.split(',')
            line_x=linedata[0] #各行のx座標 str
            line_y=linedata[1] #各行のy座標
            a[j][0]=float(line_x)
            a[j][1]=float(line_y)
            j+=1
        a = scaler.fit_transform(a)  # これを正規化するので合ってるのか？
        a = a.tolist()
        train_in, train_out = train_test_split(np.array(a), test_size=0.5, shuffle=False)  # 10*2と3*2 入力と出力
        trainX.append(train_in)
        trainY.append(train_out)
    trainX=np.array(trainX) #500*10(maxlen)*2
    trainY=np.array(trainY) #500*3*2
    # encoder,decoder
    hidden_size = 4  # 隠れ層
    encoder = Encoder(2, hidden_size,2)
    decoder = Decoder(2, hidden_size, 20) #6=2*3
    criterion = nn.MSELoss()
    encoder_optimizer = SGD(encoder.parameters(), lr=0.01)  # optimizerの初期化
    decoder_optimizer = SGD(decoder.parameters(), lr=0.01)
    # 学習開始
    batch_size = 15
    for epoch in range(240):
        running_loss = 0.0
        for i in range(int(len(trainX) / batch_size)):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            #d = torch.tensor([trainX[i]]).float()  # 入力 1*10*2 バッチ化してない方
            #label = torch.tensor([trainY[i]]).float() #出力 1*3*2
            d,label=create_batch(trainX,trainY,batch_size)
            encoder_output,encoder_state,encoder_cell=encoder(d) #エンコーダーを通過
            #ここでencoder_outputを渡したい
            encoder_hidden=(encoder_state,encoder_cell)
            decoder_hidden=encoder_hidden
            decoder_output, decoder_hidden,decoder_attention = decoder(label, decoder_hidden) #decodertensor([[0.3728, 0.1049, 0.1042]]

            label_output=[]
            for k in range(len(label)):
                a=[]
                for j in range(len(train_out)):
                    a.append(label[k][j][0].data.item())
                    a.append(label[k][j][1].data.item())
                label_output.append(a)
            label_output = torch.tensor(label_output).float()
            #print("次"+str(label[0][0][0].data.item()))
            #print("label"+str(label_output))
            loss=criterion(decoder_output,label_output)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        if (epoch % 10 == 0):
            print("loss: " + str(running_loss))
    #学習終了
    #ここからテスト
    text = "test/xy_188.txt"
    test_a = np.array([[1.0] * 2] * numline)
    f = open(text)  # ファイルを開く
    alldata = f.read()  # xy_i.txtを全部読み込む
    scaler = MinMaxScaler(feature_range=(0, 1))  # 正規化の準備
    f.close()
    lines = alldata.split('\n')  # 改行で区切る
    j = 0  # カウント用
    for line in lines:  # 1行
        linedata = line.split(',')
        line_x = linedata[0]  # 各行のx座標 str
        line_y = linedata[1]  # 各行のy座標
        test_a[j][0] = float(line_x)
        test_a[j][1] = float(line_y)
        j+=1
    test_a = scaler.fit_transform(test_a)  # これを正規化するので合ってるのか？
    test_a = test_a.tolist()
    test_in, test_out = train_test_split(np.array(test_a), test_size=0.5, shuffle=False)  # 10*2と3*2 入力と出力
    test_d = torch.tensor([test_in]).float()  #入力 1*10*2
    print(test_d.shape)
    test_label=torch.tensor([test_out]).float() #出力
    encoder_out,encoder_sta,encoder_cel=encoder(test_d) #エンコーダーを通過
    encoder_hid=(encoder_sta,encoder_cel)
    decoder_hid=encoder_hid
    decoder_out,decoder_hid,decoder_att=decoder(test_label,decoder_hid) #デコーダーを通過
    testPredict = []
    for i in range(len(decoder_out[0])):
        testPredict.append(decoder_out[0][i].data.item())
    testPredict=np.reshape(testPredict,[10,2])
    testPredict=np.array(scaler.inverse_transform(testPredict)) #正規化を元に戻す
    print(testPredict)
    # ここから accuracyの計算
    abstract = 0
    test_out = np.array(scaler.inverse_transform(test_out))
    abstract += np.sum(np.abs(testPredict - test_out))
    print(abstract/10)
    # ここまで
if __name__ == '__main__':
    main()