#coding:utf-8
#Adaboastモデルのテスト　登録してない用
import Encoder,Attention,Decoder
import adaptiveboast
import glob

import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.model_selection import train_test_split

def main():
    hidden_size = 4  # 隠れ層
    encoderstore = [] #encoderの保存用リスト
    decoderstore = [] #decoderも保存用リスト
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    trainfilenum = int(filenum * 0.8)  # 8割学習
    adabfilenum = filenum - trainfilenum
    numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt'))  #20
    for s in range(8): #loadしてlistに入れる
        encoderstore.append(torch.load('modelstore/en_model%d'%(s)))
        decoderstore.append(torch.load('modelstore/de_model%d'%(s)))

    adablist_X=[]
    adablist_Y=[]
    for s in range(1,100,1):
        text = "eattest/xy_%d.txt" % (s)
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
        test_label = torch.tensor([test_out]).float()  # 出力
        decoder_out_norm = []
        encoder2 = Encoder.Encoder(2, hidden_size, 2)
        attention2 = Attention.Attention(2, hidden_size)
        decoder2 = Decoder.Decoder(hidden_size * 2, 20)
        #出力の平均計算　予測器の数分回す
        for i in range(8):
            encoder2.load_state_dict(encoderstore[i], strict=False)
            decoder2.load_state_dict(decoderstore[i], strict=False)
            prediction_output = []  # 10回分i_number番目の出力を格納
            encoder_out, encoder_hid = encoder2(test_d)
            decoder_hid = encoder_hid
            for i_number in range(len(test_out)):
                att_concat = attention2(attention_input, decoder_hid, encoder_out)
                decoder_out = decoder2(att_concat)  # デコーダーを通過
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
                testnp = np.where(testnp == float("inf"), 0.5, testnp)
                testnp = np.where(testnp == float("-inf"), 0, testnp)
                testnp = np.where(testnp == float("nan"), 0.1, testnp)
                testnp = np.array(scaler.inverse_transform(testnp))
                testnp = np.reshape(testnp, [10, 2])
                prediction_output.append(testnp[i_number])  # i_number番目の出力を保存
            prediction_output = np.reshape(prediction_output, [10, 2])
            decoder_out_norm.append(prediction_output)
        for i in range(len(decoder_out_norm)):  # 分子のシグマ
            if (i == 0):
                decoder_value = decoder_out_norm[i]
            else:
                decoder_value += decoder_out_norm[i]
        decoder_out = decoder_value / 8
        adablist_X.append(decoder_out)
        #出力の平均計算終了
        test_out = np.array(scaler.inverse_transform(test_out))
        adablist_Y.append(test_out)

    ADE = 0.0
    FDE = 0.0
    for i in range(len(adablist_X)):
        # ADE
        ade_abstract = np.sqrt(np.square(np.abs(adablist_Y[i] - adablist_X[i])))
        ade_abstract = np.array(ade_abstract)
        ade_abstract[np.isnan(ade_abstract)] = 0
        ade_sum = (np.sum(ade_abstract)) / 10
        ADE += ade_sum
        # FDE
        fde_abstract = np.abs(adablist_Y[len(adablist_Y) - 1] - adablist_X[len(adablist_X) - 1])
        fde_abstract[np.isnan(fde_abstract)] = 0
        fde_sum = math.sqrt(np.sum(np.square(fde_abstract)))
        FDE += fde_sum
    print("ADE: " + str(ADE / len(adablist_X)))
    print("FDE: " + str(FDE / len(adablist_X)))

if __name__ == '__main__':
    main()