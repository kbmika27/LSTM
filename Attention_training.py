#coding:utf-8
#pytorchの2次元ver encoder decoder
#Attention付きLSTM 実験用
import glob
import Encoder,Decoder

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

#アテンション
class Attention(nn.Module):
    def __init__(self, inputDim, hiddenDim):  # 初期化
        super(Attention, self).__init__()
        self.hiddenDim = hiddenDim
        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True)

    def forward(self, inputs,hidden0,encoder_output):
        att_output, (hidden, cell) = self.rnn(inputs, hidden0)
        train_in = 10
        train_out = 10
        weight = np.array([[1.0] * train_in] * train_out)
        batch_concat=[]
        for batch_number in range(15):
            for l in range(train_out):  # 3
                k_t_sum = 0
                for k in range(train_in):  # 10
                    k_t = torch.dot(att_output[batch_number][l], encoder_output[batch_number][k])  # 内積
                    try:
                        k_t = math.exp(k_t.data.item())
                    except OverflowError:
                        k_t=0.0
                    weight[l][k] = k_t
                    k_t_sum += k_t
                for w in range(train_in):  # 正規化
                    weight[l][k] = weight[l][k] / k_t_sum
            cn = []
            for l in range(train_out):
                cn_sum = 0
                for k in range(train_in):
                    for h in range(4):
                        encoder_output[batch_number][k][h] = weight[l][k] * encoder_output[batch_number][k][h]
                    cn_sum += encoder_output[batch_number][k]  # m=0~10のシグマ
                cn.append(cn_sum)
            concatlist = []
            for c in range(train_out):
                concat = torch.cat([cn[c], att_output[batch_number][c]], dim=0)
                concatlist.append(concat.detach().numpy())
            batch_concat.append((concatlist))
        batch_concat=torch.tensor([batch_concat]).float()
        batch_concat=torch.reshape(batch_concat[0],[15,10,8])
        return batch_concat

#バッチ化
def create_batch(trainx, trainy, batch_size=10):
    batchX=[]
    batchY=[]
    for _ in range(batch_size):
        idx=np.random.randint(0,len(trainx)-1)
        batchX.append(trainx[idx])
        batchY.append(trainy[idx])
    batchX=np.reshape(batchX,[15,10,2])
    m=nn.BatchNorm2d(15)
    batchX=torch.tensor(batchX).float()
    batchY=torch.tensor(batchY).float()
    return batchX,batchY

def main():
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy1_data/*")  # ファイル数の取得
    filenum = len(filenum)
    numline=sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy1_data/xy_1.txt')) #13
    trainX=[]
    trainY=[]
    attention_input = []
    for i in range(filenum): #ファイル読み込み
        j=0 #カウント用
        a=np.array([[1.0] * 2] * numline)
        text="xy1_data/xy_%d.txt" % (i)
        f = open(text)  # ファイルを開く
        alldata = f.read()  # xy_i.txtを全部読み込む
        scaler = MinMaxScaler(feature_range=(0, 1)) #正規化の準備
        f.close()
        lines = alldata.split('\n') #改行で区切る
        for line in lines: #1行
            linedata =line.split(',')
            line_x=linedata[0] #各行のx座標
            line_y=linedata[1] #各行のy座標
            a[j][0]=float(line_x)
            a[j][1]=float(line_y)
            j+=1
        a = scaler.fit_transform(a)
        a = a.tolist()
        train_in, train_out = train_test_split(np.array(a), test_size=0.5, shuffle=False)  # 10*2と3*2 入力と出力
        attention_in = []
        for _ in range(len(train_in)):
            attention_in.append(train_in[len(train_in) - 1])
        attention_input.append(attention_in)
        trainX.append(train_in)
        trainY.append(train_out)
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    attention_input = np.array(attention_input)

    hidden_size = 4  # 隠れ層
    encoder = Encoder.Encoder(2, hidden_size,2)
    attention=Attention(2,hidden_size)
    decoder = Decoder.Decoder(hidden_size*2, 20)
    criterion = nn.MSELoss()
    encoder_optimizer = SGD(encoder.parameters(), lr=0.01)  # optimizerの初期化
    attention_optimizer = SGD(attention.parameters(), lr=0.01)
    decoder_optimizer = SGD(decoder.parameters(), lr=0.01)
    # 学習開始
    batch_size = 15
    for epoch in range(15):
        running_loss = 0.0
        for i in range(int(len(trainX) / batch_size)):
            encoder_optimizer.zero_grad()
            attention_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            #d = torch.tensor([trainX[i]]).float()  # 入力
            #label = torch.tensor([trainY[i]]).float() #出力 　バッチ化しない時に利用
            d, attention_label = create_batch(trainX, attention_input, batch_size)
            label, nonuse = create_batch(trainY, trainY, batch_size)
            encoder_output,encoder_hidden=encoder(d) #エンコーダーを通過
            decoder_hidden = encoder_hidden
            # ここから attention
            train_out_batch = np.array([[[1.0] * 2] * 10] * batch_size)
            for i_number in range(len(train_out)):
                concat = attention(attention_label, decoder_hidden, encoder_output)
                decoder_output = decoder(concat)
                valuesame_batch = []
                for batch_number in range(15):
                    train_out_batch[batch_number][i_number][0] = decoder_output[batch_number][i_number * 2].data.item()
                    train_out_batch[batch_number][i_number][1] = decoder_output[batch_number][i_number * 2 + 1].data.item()
                    valuesame = []  # i番目を伸ばしたlist
                    for _ in range(len(train_out)):
                        valuesame.append(decoder_output[batch_number][i_number * 2].data.item())
                        valuesame.append(decoder_output[batch_number][i_number * 2 + 1].data.item())
                    valuesame_batch.append(valuesame)
                attention_label = valuesame_batch
                attention_label = torch.tensor(attention_label).float()
                attention_label = torch.reshape(attention_label, [batch_size, 10, 2])
            #ここまで　attention
            label = torch.tensor(label).float()
            train_out_batch = torch.tensor(train_out_batch, requires_grad=True).float()
            loss = criterion(label, train_out_batch)
            loss.backward()
            encoder_optimizer.step()
            attention_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        print('%d loss: %.3f' % (epoch + 1, running_loss))
    torch.save(encoder.state_dict(), 'att_en_model' )
    torch.save(decoder.state_dict(), 'att_de_model' )
    print("学習終了")


if __name__ == '__main__':
    main()