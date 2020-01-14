#coding:utf-8
#信頼度を用いてテストをする
#登録済みの人用
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
    personnum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/label_reliability/*")  # ファイル数を取得する
    personnum = len(personnum)
    hidden_size = 4  # 隠れ層
    encoderstore = [] #encoderの保存用リスト
    decoderstore = [] #decoderも保存用リスト
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    trainfilenum = int(filenum * 0.8)  # 8割学習
    adabfilenum = filenum - trainfilenum
    numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt'))  # 13
    for s in range(3): #loadしてlistに入れる
        encoderstore.append(torch.load('en_model%d'%(s)))
        decoderstore.append(torch.load('de_model%d'%(s)))
    #mainmethod=adaptiveboast.Main()
    #label_reliability=mainmethod.main()
    readtext = "label_reliability/data0.npy"
    label_reliability=(np.load(readtext)).tolist()
    print(label_reliability)
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

    # ここから出力の平均を取る
    decoder_out_norm = []  # 分子10*2*2を入れておくlist
    for i in range(len(label_reliability)):  # 出力の平均をとるfor文 2
        encoder2 = Encoder.Encoder(2, hidden_size, 2)
        attention2 = Attention.Attention(2, hidden_size)
        decoder2 = Decoder.Decoder(hidden_size * 2, 20)  # 6=2*3
        # encoder2.load_state_dict(torch.load('en_model'),strict=False)
        encoder2.load_state_dict(encoderstore[int(label_reliability[i][0])], strict=False)
        encoder_out, encoder_hid = encoder2(test_d)
        decoder_hid = encoder_hid
        att_concat = attention2(test_label, decoder_hid, encoder_out)
        # decoder2.load_state_dict(torch.load('de_model'),strict=False)
        decoder2.load_state_dict(decoderstore[0], strict=False)
        decoder_out = decoder2(att_concat)  # デコーダーを通過
        testnp = []  # lstmからの出力1*20を入れるlist
        for j in range(len(decoder_out[0])):  # 20
            testnp.append(decoder_out[0][j].data.item())  # 正規化されていない
        testnp = np.reshape(testnp, [10, 2])
        testnp = np.array(scaler.inverse_transform(testnp))  # 正規化を元に戻す
        print("testnp" + str(testnp))
        testnp = [g * math.log(1 / label_reliability[i][1]) for g in testnp]  # 各出力結果にlog(1/信頼度)をかける
        testnp = np.reshape(testnp, [10, 2])
        decoder_out_norm.append(testnp)  # len2
    for i in range(len(decoder_out_norm)):  # 分子のシグマ
        if (i == 0):
            decoder_value = decoder_out_norm[i]
        else:
            decoder_value += decoder_out_norm[i]
    decoder_out = decoder_value
    value = 0.0
    for i in range(len(decoder_out_norm)):  # 分母のシグマ
        # print("分母"+str(math.log(1/label_reliability[i][1])))
        value += math.log(1 / label_reliability[i][1])  # 1/信頼度を足す
    decoder_out = decoder_out / value
    print("deoder" + str(decoder_out))
    # ここまでで出力の平均計算終了

    # ここから accuracyの計算
    abstract = 0
    test_out = np.array(scaler.inverse_transform(test_out))
    print("正解" + str(test_out))
    abstract += np.sum(np.abs(decoder_out - test_out))
    print(abstract / 10)
    # ここまで

if __name__ == '__main__':
    main()