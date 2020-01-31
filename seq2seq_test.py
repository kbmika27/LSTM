#coding:utf-8
#sequence to sequence テスト用
import glob
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
#エンコーダー
class Encoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Encoder,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
    def forward(self, inputs, hidden0=None):
        output,(state, cell) = self.rnn(inputs, hidden0)
        return output,state,cell
#デコーダー
class Decoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):  #初期化
        super(Decoder,self).__init__()
        self.hiddenDim=hiddenDim
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim) #全結合
    def forward(self, inputs, hidden0):
        output,(hidden,cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])
        return output,(hidden,cell)

def main():
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/*")  # ファイル数の取得
    filenum = len(filenum)
    numline=sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt')) #13
    seq_encoder = torch.load('modelstore/seq2_en_model')
    seq_decoder = torch.load('modelstore/seq2_de_model')
    hidden_size = 4  # 隠れ層
    encoder = Encoder(2, hidden_size,2)
    decoder = Decoder(2, hidden_size, 20)
    encoder.load_state_dict(seq_encoder, strict=False)
    decoder.load_state_dict(seq_decoder, strict=False)
    seq_listX=[] #正解データ格納list
    seq_listY=[] #出力結果格納list

    #ここからテスト
    for s in range(12,13,1):
        text = "gardentxt/mika%d.txt" % (s)  #テストに使うファイル名
        test_a = np.array([[1.0] * 2] * numline)
        f = open(text)  # ファイルを開く
        alldata = f.read()  # xy_i.txt読み込み
        scaler = MinMaxScaler(feature_range=(0, 1))
        f.close()
        lines = alldata.split('\n')
        j = 0  # カウント用
        for line in lines:  # 1行
            linedata = line.split(',')
            line_x = linedata[0]  # 各行のx座標
            line_y = linedata[1]  # 各行のy座標
            test_a[j][0] = float(line_x)
            test_a[j][1] = float(line_y)
            j += 1
        test_a = scaler.fit_transform(test_a)
        test_a = test_a.tolist()
        test_in, test_out = train_test_split(np.array(test_a), test_size=0.5, shuffle=False)  # 10*2と3*2 入力と出力
        seq_input = []
        for i in range(len(test_in)):
            seq_input.append(test_in[len(test_in) - 1])
        seq_input = torch.tensor([seq_input]).float()
        seq_input = torch.reshape(seq_input, [1, 10, 2])
        test_d = torch.tensor([test_in]).float()
        test_label = torch.tensor([test_out]).float()
        encoder_out, encoder_sta, encoder_cel = encoder(test_d)  # エンコーダーを通過
        encoder_hid = (encoder_sta, encoder_cel)
        decoder_hid = encoder_hid
        prediction_output = []  # 10回分i_number番目の出力を格納
        for i_number in range(len(test_out)):
            decoder_out, decoder_hid = decoder(test_label, decoder_hid)  # デコーダーを通過
            valuesame = []  # i_number番目を伸ばしてattentionに入れる
            for _ in range(len(test_out)):  #
                valuesame.append(decoder_out[0][i_number * 2].data.item())
                valuesame.append(decoder_out[0][i_number * 2 + 1].data.item())
            valuesame = torch.tensor([valuesame])
            valuesame = torch.reshape(valuesame, [1, 10, 2])
            seq_input = valuesame
            testnp = []  # lstmからの出力1*20を入れるlist
            for j in range(len(decoder_out[0])):
                testnp.append(decoder_out[0][j].data.item())
            testnp = np.reshape(testnp, [10, 2])
            testnp = np.array(scaler.inverse_transform(testnp))
            testnp = np.reshape(testnp, [10, 2])
            prediction_output.append(testnp[i_number])  # i_number番目の出力を保存
        prediction_output = np.array(prediction_output)
        seq_listY.append(prediction_output)
        test_out = np.array(scaler.inverse_transform(test_out))
        seq_listX.append(test_out)
    ADE = 0
    FDE = 0.0
    for i in range(len(seq_listX)):
        # ADE
        ade_abstract = np.sqrt(np.square(np.abs(seq_listY[i]- seq_listX[i])))
        ade_sum = (np.sum(ade_abstract)) / 10
        ADE+=ade_sum
        # FDE
        fde_abstract = np.abs(seq_listY[len(seq_listY) - 1] - seq_listX[len(seq_listX) - 1])
        fde_sum = math.sqrt(np.sum(np.square(fde_abstract)))
        FDE+=fde_sum
    print("ADE: " + str(ADE/len(seq_listX)))
    print("FDE: " + str(FDE/len(seq_listY)))

if __name__ == '__main__':
    main()