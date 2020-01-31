#coding:utf-8
#Attention付きLSTM テスト用
import glob
import Encoder,Decoder,Attention
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split


def main():
    hidden_size = 4  # 隠れ層
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/*")  # ファイル数の取得
    filenum = len(filenum)
    trainfilenum = int(filenum * 0.8)
    adabfilenum = filenum - trainfilenum
    numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt'))  # 20
    att_encoder=torch.load('modelstore/att_en_model')
    att_decoder=torch.load('modelstore/att_de_model')

    #ここからテスト
    attlist_X=[]
    attlist_Y=[]
    for s in range(12,13,1):
        text = "gardentxt/mika%d.txt" % (s)
        test_a = np.array([[1.0] * 2] * numline)
        f = open(text)
        alldata = f.read()
        scaler = MinMaxScaler(feature_range=(0, 1))  # 正規化の準備
        f.close()
        lines = alldata.split('\n')  # 改行で区切る
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
        test_in, test_out = train_test_split(np.array(test_a), test_size=0.5, shuffle=False)
        attention_input = []  # 1回目はencoder入力の最終段を格納
        for i in range(len(test_in)):
            attention_input.append(test_in[len(test_in) - 1])
        attention_input = torch.tensor([attention_input]).float()
        attention_input = torch.reshape(attention_input, [1, 10, 2])
        test_d = torch.tensor([test_in]).float()
        encoder = Encoder.Encoder(2, hidden_size, 2)
        attention = Attention.Attention(2, hidden_size)
        decoder = Decoder.Decoder(hidden_size * 2, 20)
        encoder.load_state_dict(att_encoder, strict=False)
        decoder.load_state_dict(att_decoder, strict=False)
        test_label = torch.tensor([test_out]).float()
        encoder_out, encoder_hid = encoder(test_d) #encoder通過
        decoder_hid = encoder_hid
        prediction_output = []  # 10回分i_number番目の出力を格納
        #attention内
        for i_number in range(len(test_out)):
            att_concat = attention(attention_input, decoder_hid, encoder_out)
            decoder_out = decoder(att_concat)  #decoder通過
            valuesame = []  # i_number番目を伸ばしてattentionに入れる
            for _ in range(len(test_out)):  #
                valuesame.append(decoder_out[0][i_number * 2].data.item())
                valuesame.append(decoder_out[0][i_number * 2 + 1].data.item())
            valuesame = torch.tensor([valuesame])
            valuesame = torch.reshape(valuesame, [1, 10, 2])
            attention_input = valuesame
            testnp = []  # lstmからの出力1*20を入れるlist
            for j in range(len(decoder_out[0])):  # 20
                testnp.append(decoder_out[0][j].data.item())
            testnp = np.reshape(testnp, [10, 2])
            for width in range(2):
                for height in range(10):
                    if (testnp[height][width] == float("inf") or testnp[height][width] == float("-inf")):
                        testnp[height][width] = 0.1
            testnp = np.array(scaler.inverse_transform(testnp))  # 正規化を元に戻す
            testnp = np.reshape(testnp, [10, 2])
            prediction_output.append(testnp[i_number])  # i_number番目の出力を保存
        prediction_output = np.array(prediction_output)
        attlist_Y.append(prediction_output)
        test_out = np.array(scaler.inverse_transform(test_out))
        attlist_X.append(test_out)
    ADE = 0.0
    FDE = 0.0
    print(len(attlist_X))
    for i in range(len(attlist_X)):
        # ADE
        ade_abstract = np.sqrt(np.square(np.abs(attlist_Y[i] - attlist_X[i])))
        ade_abstract=np.array(ade_abstract)
        ade_abstract=np.where(ade_abstract>500,200,ade_abstract)
        ade_abstract[np.isnan(ade_abstract)]=0
        ade_sum = (np.sum(ade_abstract)) / 10
        ADE += ade_sum
        # FDE
        fde_abstract = np.abs(attlist_Y[len(attlist_Y) - 1] - attlist_X[len(attlist_X) - 1])
        fde_abstract[np.isnan(fde_abstract)] = 0.5
        fde_abstract=np.where(fde_abstract>500,200,fde_abstract)
        fde_sum = math.sqrt(np.sum(np.square(fde_abstract)))
        FDE += fde_sum
    print("ADE: " + str(ADE / len(attlist_X)))
    print("FDE: " + str(FDE / len(attlist_X)))
if __name__ == '__main__':
    main()