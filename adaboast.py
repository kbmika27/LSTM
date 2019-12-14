#coding:utf-8
#pytorchの2次元ver encoder decoder attention
import glob

import pandas as pd
import random
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
    encoderstore = [] #encoderの保存用リスト
    decoderstore = [] #decoderも保存用リスト
    for s in range(3): #xy(i)_dataをs回ループ
        filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy%d_data/*"%(s))  # ファイル数を取得する
        filenum = len(filenum)
        trainfilenum=int(filenum*0.8) #8割学習
        adabfilenum=filenum-trainfilenum
        numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy%d_data/xy_0.txt'%(s)))  # 13
        trainX = []
        trainY = []
        for i in range(trainfilenum):  # ファイルの読み込み
            j = 0  # カウント用
            a = np.array([[1.0] * 2] * numline)
            text = "xy%d_data/xy_%d.txt" % (s,i)
            f = open(text)  # ファイルを開く
            alldata = f.read()  # xy_i.txtを全部読み込む
            scaler = MinMaxScaler(feature_range=(0, 1))  # 正規化の準備
            f.close()
            lines = alldata.split('\n')  # 改行で区切る
            for line in lines:  # 1行
                linedata = line.split(',')
                line_x = linedata[0]  # 各行のx座標 str
                line_y = linedata[1]  # 各行のy座標
                a[j][0] = float(line_x)
                a[j][1] = float(line_y)
                j += 1
            a = scaler.fit_transform(a)  # これを正規化するので合ってるのか？
            a = a.tolist()
            train_in, train_out = train_test_split(np.array(a), test_size=0.5, shuffle=False)  # 10*2と3*2 入力と出力
            trainX.append(train_in)
            trainY.append(train_out)
        trainX = np.array(trainX)  # 500*10(maxlen)*2
        trainY = np.array(trainY)  # 500*3*2

        # encoder,decoder
        hidden_size = 4  # 隠れ層
        encoder = Encoder(2, hidden_size, 2)
        attention = Attention(2, hidden_size)
        decoder = Decoder(hidden_size * 2, 20)  # 6=2*3
        criterion = nn.MSELoss()
        encoder_optimizer = SGD(encoder.parameters(), lr=0.01)  # optimizerの初期化
        decoder_optimizer = SGD(decoder.parameters(), lr=0.01)
        # 学習開始
        batch_size = 15
        for epoch in range(15):
            running_loss = 0.0
            for i in range(int(len(trainX) / batch_size)):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                # d = torch.tensor([trainX[i]]).float()  # 入力 1*10*2 バッチ化してない方
                # label = torch.tensor([trainY[i]]).float() #出力 1*3*2
                d, label = create_batch(trainX, trainY, batch_size)
                # ここでバッチ正則化
                encoder_output, encoder_hidden = encoder(d)  # エンコーダーを通過
                decoder_hidden = encoder_hidden
                concat = attention(label, decoder_hidden, encoder_output)  # アテンションを通過
                decoder_output = decoder(concat)  # decodertensor([[0.3728, 0.1049, 0.1042]]]
                label_output = []
                for k in range(len(label)):
                    a = []
                    for j in range(len(train_out)):
                        a.append(label[k][j][0].data.item())
                        a.append(label[k][j][1].data.item())
                    label_output.append(a)
                label_output = torch.tensor(label_output).float()
                # print("次"+str(label[0][0][0].data.item()))
                # print("label"+str(label_output))
                loss = criterion(decoder_output, label_output)
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                running_loss += loss.item()
            if (epoch % 10 == 0):
                print("loss: " + str(running_loss))
        encoderstore.append(encoder.state_dict())
        decoderstore.append(decoder.state_dict())
        #torch.save(encoder.state_dict(),'en_model')  #学習済みモデルを保存
        #torch.save(decoder.state_dict(),'de_model')
        #print("model"+str(encoderhidden))
        # 学習終了
    #ここからadaboost
    adabX=[] #入力
    adabY=[] #出力
    weightlist=[] #重みを保存するlist
    cumulativeweight=[] #累積確率を格納するlist
    adab_weight=[] #重みにラベルをつけたlist
    weight=0.0
    label_reliability=[]
    for s in range(3):
        for i in range(trainfilenum, filenum, 1):
            j = 0  # カウント用
            a = np.array([[1.0] * 2] * numline)
            text = "xy%d_data/xy_%d.txt" % (s, i)
            f = open(text)  # ファイルを開く
            alldata = f.read()  # xy_i.txtを全部読み込む
            scaler = MinMaxScaler(feature_range=(0, 1))  # 正規化の準備
            f.close()
            lines = alldata.split('\n')  # 改行で区切る
            for line in lines:  # 1行
                linedata = line.split(',')
                line_x = linedata[0]  # 各行のx座標 str
                line_y = linedata[1]  # 各行のy座標
                a[j][0] = float(line_x)
                a[j][1] = float(line_y)
                j += 1
            a = scaler.fit_transform(a)  # これを正規化するので合ってるのか？
            a = a.tolist()
            adbtrain_in, adbtrain_out = train_test_split(np.array(a), test_size=0.5, shuffle=False)  # 10*2と3*2 入力と出力
            adabX.append([adbtrain_in]) #入力の2*10が200
            adabY.append(adbtrain_out)
            adab_weight.append(s) #ラベルを重みにつけてる
    for i in range(len(adabX)):
        weight =1.0/len(adabX)  # 重みの初期値
        weightlist.append(weight)
    probability=0 #確率
    for epoch in range(2):#何個選ぶか
        errorlist = []  # 各LSTMのエラー率を保存するlist
        cumulativeweight = []  # 累積確率を格納するlist
        p_denominator = 0  # 分母
        for i in range(len(adabX)):  # 分母を足すfor文
            p_denominator += weightlist[i]
        for i in range(len(adabX)):  # 分子/分母をしてlistに入れるfor文
            p_molecule = weightlist[i]  # 分子
            probability = p_molecule / p_denominator
            weightlist[i] = probability
        value = 0.0
        for i in range(len(adabX)):  # 累積確率のlist
            value += weightlist[i]
            cumulativeweight.append(value)
        p_random = random.uniform(0, 1)  # この値を使って選ぶ
        i = 0  # ラベル
        while i < len(cumulativeweight):
            if (cumulativeweight[i] < p_random):
                i += 1
            else:  # 大きくなった瞬間に抜ける
                break
        p_label = adab_weight[i]  # 0~Mの値を取得
        print(p_label)
        # エラー計算 これを各lstmに対してやる
        for t in range(len(adabX)):
            adab_d = torch.tensor(adabX[t]).float()  # 入力 1*10*2
            adab_label = torch.tensor([adabY[t]]).float()  # 出力
            encoder2 = Encoder(2, hidden_size, 2)
            decoder2 = Decoder(hidden_size * 2, 20)  # 6=2*3
            encoder2.load_state_dict(encoderstore[p_label], strict=False)
            encoder_out, encoder_hid = encoder2(adab_d)
            decoder_hid = encoder_hid
            att_concat = attention(adab_label, decoder_hid, encoder_out)
            decoder2.load_state_dict(decoderstore[p_label], strict=False)
            decoder_out = decoder2(att_concat)  # デコーダーを通過
            decoder_out_np=[]#デコーダーから出てきたものを正規化元に戻す
            for i in  range(len(decoder_out[0])):
                decoder_out_np.append(decoder_out[0][i].data.item())
            decoder_out_np = np.reshape(decoder_out_np, [10, 2])  # 出力をnpに入れたもの
            decoder_out_np = np.array(scaler.inverse_transform(decoder_out_np))  # 正規化を元に戻す
            adabY[t]=scaler.inverse_transform(adabY[t])
            errornp=np.abs(decoder_out_np-adabY[t])
            error=sum(errornp)#20個の誤差の合計
            error=sum(error)
            errorlist.append(error)
        print(len(errorlist))
        error_rate = 0.0  # エラー率
        print("誤差" + str(errorlist))
        print(len(errorlist))
        for i in range(len(errorlist)):
            if (errorlist[i] > 300):
                error_rate += weightlist[i]
        error_reliability = error_rate * error_rate
        print(error_reliability)
        weightlist_sum = 0.0
        for i in range(len(weightlist)):  # 重みの更新
            if (errorlist[i] <= 800):
                weightlist[i] = weightlist[i] * error_reliability
            else:
                weightlist[i] = 1
            weightlist_sum += weightlist[i]
        for i in range(len(weightlist)):
            weightlist[i] = weightlist[i] / weightlist_sum
        label_reliability.append([p_label,error_reliability])
    print(label_reliability)

    #ここまでadaboast
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

    # ここから出力の平均を取る
    decoder_out_norm=[]#分子を入れておくlist
    for i in range(len(label_reliability)): #出力の平均をとるfor文
        encoder2 = Encoder(2, hidden_size, 2)
        decoder2 = Decoder(hidden_size * 2, 20)  # 6=2*3
        # encoder2.load_state_dict(torch.load('en_model'),strict=False)
        encoder2.load_state_dict(encoderstore[label_reliability[i][0]], strict=False)
        encoder_out, encoder_hid = encoder2(test_d)
        decoder_hid = encoder_hid
        att_concat = attention(test_label, decoder_hid, encoder_out)
        # decoder2.load_state_dict(torch.load('de_model'),strict=False)
        decoder2.load_state_dict(decoderstore[0], strict=False)
        decoder_out = decoder2(att_concat)  # デコーダーを通過
        testnp=[] #lstmの予測1*20を入れるlist
        for j in range(len(decoder_out[0])):
            testnp.append(decoder_out[0][j].data.item())
        testnp= [g*math.log(1/label_reliability[i][1]) for g in testnp] #各出力結果にlog(1/信頼度)をかける
        testnp = np.reshape(testnp, [10, 2])
        testnp = np.array(scaler.inverse_transform(testnp))  # 正規化を元に戻す
        print(testnp)
        decoder_out_norm.append(testnp)
    print("norm"+str(decoder_out_norm))
    for i in range(len(decoder_out_norm)): #分子のシグマ
        if(i==0):
            decoder_value=decoder_out_norm[i]
        else:
            decoder_value+=decoder_out_norm[i]
    decoder_out=decoder_value
    print("10*2で出て欲しい"+str(decoder_out.shape))
    for i in range(len(decoder_out_norm)):#分母のシグマ
        value=0.0
        print("分母"+str(math.log(1/label_reliability[i][1])))
        value+=math.log(1/label_reliability[i][1])
    decoder_out=decoder_out/value
    print(decoder_out)
    #ここまでで出力の平均計算終了

    # ここから accuracyの計算
    abstract = 0
    test_out = np.array(scaler.inverse_transform(test_out))
    print("正解"+str(test_out))
    abstract += np.sum(np.abs(decoder_out - test_out))
    print(abstract/10)
    # ここまで

if __name__ == '__main__':
    main()