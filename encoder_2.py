#coding:utf-8
#pytorchの2次元ver encoder decoder attention
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

#エンコーダー 変えてないから動くはず
class Encoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Encoder,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)

    def forward(self, inputs, hidden0=None): #予測をする
        output,(state, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル
        return output,(state,cell)

#アテンション
class Attention(nn.Module):
    def __init__(self, inputDim, hiddenDim):  # 初期化
        super(Attention, self).__init__()
        self.hiddenDim = hiddenDim
        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True)

    def forward(self, inputs,hidden0,encoder_output):
        att_output, (hidden, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル ここには全部コピーを入れる
        train_in=10
        train_out=10
        weight = np.array([[1.0] * train_in] * train_out)  # 3*10 an(n)を格納
        for l in range(train_out):  # 3
            k_t_sum = 0  # 正規化するため分子を足す
            for k in range(train_in):  # 10
                t = att_output[0][l].t()  # デコーダーの転置
                k_t = torch.dot(t, encoder_output[0][k])  # 内積
                k_t = math.exp(k_t.data.item())
                weight[l][k] = k_t
                k_t_sum += k_t
            for w in range(train_in):  # 正規化
                weight[l][k] = weight[l][k] / k_t_sum
        cn = []  # 4*1が3つ格納されるはず c0 c1 c2
        for l in range(train_out):  # 3
            cn_sum = 0
            for k in range(train_in):  # 10
                for h in range(4):
                    encoder_output[0][k][h] = weight[l][k] * encoder_output[0][k][h]
                cn_sum += encoder_output[0][k]  # m=0~10のシグマ
            cn.append(cn_sum)  # .data付けても付けなくてもOK!
        #print("cn"+str(cn))
        concatlist=[]
        for c in range(train_out):
            concat=torch.cat([cn[c], att_output[0][c]], dim=0) #8*1のtensor型
            concatlist.append(concat.detach().numpy())
        concatlist=torch.tensor([concatlist]).float()
        return concatlist

#デコーダー
class Decoder(nn.Module):
    def __init__(self,hiddenDim,outputDim):  #初期化
        super(Decoder,self).__init__()
        self.bn1=nn.BatchNorm1d(outputDim) #バッチ正規化
        self.output_layer = nn.Linear(hiddenDim, outputDim)  # 全結合層

    def forward(self, concat): #予測をする
        output = self.output_layer(concat[:, -1, :])
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
    #print(batchX)
    batchX=np.reshape(batchX,[15,10,2])
    #print(batchX)
    m=nn.BatchNorm2d(15)
    batchX=torch.tensor(batchX).float()
    batchY=torch.tensor(batchY).float()
    return batchX,batchY

def main():
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy1_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    numline=sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy1_data/xy_1.txt')) #13
    trainX=[]
    trainY=[]
    for i in range(filenum): #ファイルの読み込み
        j=0 #カウント用
        a=np.array([[1.0] * 2] * numline)
        text="xy1_data/xy_%d.txt" % (i)
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
    attention=Attention(2,hidden_size)
    decoder = Decoder(hidden_size*2, 20) #6=2*3
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
            #d = torch.tensor([trainX[i]]).float()  # 入力 1*10*2 バッチ化してない方
            #label = torch.tensor([trainY[i]]).float() #出力 1*3*2
            d,label=create_batch(trainX,trainY,batch_size)
            #ここでバッチ正則化
            encoder_output,encoder_hidden=encoder(d) #エンコーダーを通過
            decoder_hidden = encoder_hidden
            concat=attention(label,decoder_hidden,encoder_output) #アテンションを通過
            decoder_output= decoder(concat) #decodertensor([[0.3728, 0.1049, 0.1042]]
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
    text = "test/xy_0.txt"
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
    encoder_out,encoder_hid=encoder(test_d) #エンコーダーを通過
    decoder_hid=encoder_hid
    att_concat=attention(test_label,decoder_hid,encoder_out)
    decoder_out=decoder(att_concat) #デコーダーを通過
    testPredict = []
    for i in range(len(decoder_out[0])):
        testPredict.append(decoder_out[0][i].data.item())
    testPredict=np.reshape(testPredict,[10,2]) #出力をnpに入れたもの
    testPredict=np.array(scaler.inverse_transform(testPredict)) #正規化を元に戻す
    print(testPredict)
    # ここから accuracyの計算
    abstract = 0
    test_out = np.array(scaler.inverse_transform(test_out))
    abstract += np.sum(np.abs(testPredict - test_out))
    print("正解"+str(test_out))
    print(abstract/10)
    # ここまで

if __name__ == '__main__':
    main()