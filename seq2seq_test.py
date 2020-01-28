#coding:utf-8
#sequence to sequence 実験の時に使う
import glob
import math
import numpy as np
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
        output = self.output_layer(output[:, -1, :])
        return output,(hidden,cell)

def main():
    #ここから
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    numline=sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt')) #13
    # encoder,decoder
    seq_encoder = torch.load('modelstore/seq2_en_model')
    seq_decoder = torch.load('modelstore/seq2_de_model')
    hidden_size = 4  # 隠れ層
    encoder = Encoder(2, hidden_size,2)
    decoder = Decoder(2, hidden_size, 20) #6=2*3
    encoder.load_state_dict(seq_encoder, strict=False)
    decoder.load_state_dict(seq_decoder, strict=False)

    #ここからテスト
    text = "test/xy_1.txt"
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
    seq_input = []  # 1回目はencoder入力の最終段を格納
    for i in range(len(test_in)):  # attention_inputに最初に入力するlist
        seq_input.append(test_in[len(test_in) - 1])
    seq_input = torch.tensor([seq_input]).float()
    seq_input = torch.reshape(seq_input, [1, 10, 2])  # 1*10*2
    test_d = torch.tensor([test_in]).float()  #入力 1*10*2
    print(test_d.shape)
    test_label=torch.tensor([test_out]).float() #出力
    encoder_out,encoder_sta,encoder_cel=encoder(test_d) #エンコーダーを通過
    encoder_hid=(encoder_sta,encoder_cel)
    decoder_hid=encoder_hid
    #decoderからの出力をぐるぐる回す
    prediction_output = []  # 10回分i_number番目の出力を格納
    for i_number in range(len(test_out)):
        decoder_out, decoder_hid= decoder(test_label, decoder_hid)  # デコーダーを通過
        valuesame = []  # i_number番目を伸ばしてattentionに入れる
        for _ in range(len(test_out)):  #
            valuesame.append(decoder_out[0][i_number * 2].data.item())
            valuesame.append(decoder_out[0][i_number * 2 + 1].data.item())
        valuesame = torch.tensor([valuesame])
        valuesame = torch.reshape(valuesame, [1, 10, 2])
        seq_input = valuesame
        testnp = []  # lstmからの出力1*20を入れるlist
        for j in range(len(decoder_out[0])):  # 20
            testnp.append(decoder_out[0][j].data.item())  # 正規化されていない
        testnp = np.reshape(testnp, [10, 2])
        testnp = np.array(scaler.inverse_transform(testnp))  # 正規化を元に戻す
        testnp = np.reshape(testnp, [10, 2])
        prediction_output.append(testnp[i_number])  # i_number番目の出力を保存
    prediction_output = np.array(prediction_output)
    print(prediction_output)
    abstract = 0
    test_out = np.array(scaler.inverse_transform(test_out))
    # ADEの計算 2嬢　ルート　たす　わる
    ade_abstract = np.sqrt(np.square(np.abs(prediction_output - test_out)))
    ade_sum = (np.sum(ade_abstract)) / 10
    print("ADE: " + str(ade_sum))

    # FDE
    fde_abstract = np.abs(prediction_output[len(prediction_output) - 1] - test_out[len(test_out) - 1])
    fde_sum = math.sqrt(np.sum(np.square(fde_abstract)))
    print("FDE: " + str(fde_sum))

if __name__ == '__main__':
    main()