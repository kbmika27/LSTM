#coding:utf-8
#学習済モデルを呼び出し、adaboastの学習をし、ラベルと信頼度を返す
import Encoder,Attention,Decoder
import glob

import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.optim import SGD
from sklearn.model_selection import train_test_split



def errorcalculate(errorlist,weightlist): # エラー率の計算
    error_rate = 0.0  # エラー率
    for i in range(len(errorlist)):
        if (errorlist[i] > 5):
            error_rate += weightlist[i]
    return error_rate

def weightcalculate(errorlist,weightlist,error_reliability): #重みの計算
    weightlist_sum = 0.0
    for i in range(len(weightlist)):
        if (errorlist[i] <= 5):
            weightlist[i] = weightlist[i] * error_reliability
        else:
            weightlist[i] = weightlist[i] * 1
        weightlist_sum += weightlist[i]
    for j in range(len(weightlist)):
        weightlist[j] = weightlist[j] / (weightlist_sum)
    return weightlist

class Main:
    def main(self):
        hidden_size = 4  # 隠れ層
        encoderstore = []  # encoderの保存用リスト
        decoderstore = []  # decoderも保存用リスト
        filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/person0/*")  # ファイル数を取得する 200
        filenum = len(filenum)
        trainfilenum = int(filenum * 0.8)  # 8割学習 160
        adabfilenum = filenum - trainfilenum #40
        numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt'))  # 13
        for s in range(3):  # loadしてlistに入れる
            encoderstore.append(torch.load('en_model%d' % (s)))
            decoderstore.append(torch.load('de_model%d' % (s)))
        # ここからadaboost
        adabX = []  # 入力　adaboost用の学習データ
        adabY = []  # 出力
        weightlist = []  # 重みを保存するlist
        label_reliability = []  # 信頼度を入れるlist
        for i in range(filenum): #personiのadab用のデータ作り
            j = 0  # カウント用
            a = np.array([[1.0] * 2] * numline)
            text = "person0/xy_%d.txt" % (i)
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
            adbtrain_in, adbtrain_out = train_test_split(np.array(a), test_size=0.5,
                                                         shuffle=False)  # 10*2と3*2 入力と出力
            adabX.append([adbtrain_in])  # 入力の2*10が120
            adabY.append(adbtrain_out)
        for i in range(len(adabX)):
            weight = 1.0 / len(adabX)  # 重みの初期値
            weightlist.append(weight)
        #ここまでは合ってる
        for epoch in range(2):  # 何個選ぶか T個
            error_rate_list = []  # 各lstmのエラー率を入れる
            attention2 = Attention.Attention(2,hidden_size)
            print("あ"+str(weightlist))
            # エラー計算 これを各lstmに対してやる
            for p_label in range(3):
                errorlist = []  # 各LSTMのエラー率を保存するlist
                print(str(p_label)+"回目")
                for t in range(len(adabX)):  # 300
                    adab_d = torch.tensor(adabX[t]).float()  # 入力 1*10*2
                    adab_label = torch.tensor([adabY[t]]).float()  # 出力
                    encoder2 = Encoder.Encoder(2, hidden_size, 2)
                    decoder2 = Decoder.Decoder(hidden_size * 2, 20)  # 6=2*3
                    encoder2.load_state_dict(encoderstore[p_label], strict=False)  # encoder通過
                    encoder_out, encoder_hid = encoder2(adab_d)
                    decoder_hid = encoder_hid
                    att_concat = attention2(adab_label, decoder_hid, encoder_out)  # attention通過
                    decoder2.load_state_dict(decoderstore[p_label], strict=False)
                    decoder_out = decoder2(att_concat)  # デコーダーを通過
                    decoder_out_np = []  # デコーダーから出てきたものを正規化元に戻すためのlist
                    for i in range(len(decoder_out[0])):
                        decoder_out_np.append(decoder_out[0][i].data.item())
                    decoder_out_np = np.reshape(decoder_out_np, [10, 2])  # 出力をnpに入れたもの
                    decoder_out_np = np.array(scaler.inverse_transform(decoder_out_np))  # 正規化を元に戻す
                    adabY[t] = scaler.inverse_transform(adabY[t])  # 正解データ
                    errornp = np.abs((decoder_out_np - adabY[t]) / adabY[t])  # 正解と出力の差/正解
                    error = sum(errornp)  # 20個の誤差の合計
                    error = sum(error)
                    errorlist.append(error)
                print("誤差" + str(errorlist))
                print(len(errorlist))
                error_rate=errorcalculate(errorlist,weightlist) #エラー率の計算
                error_rate_list.append(error_rate)
            min_label=error_rate_list.index(min(error_rate_list)) #エラー率が最小になるラベルを取得
            error_reliability=float(min(error_rate_list))*float(min(error_rate_list)) #エラー率が最小になるラベルの信頼度計算
            print("い"+str(min(error_rate_list)))
            print("error list"+str(error_rate_list))
            weightlist=weightcalculate(errorlist,weightlist,error_reliability) #重みの更新
            label_reliability.append([min_label,error_reliability]) #ラベルと信頼度を格納
        label_reliability=np.array(label_reliability)
        print(np.array(label_reliability))
        filename="label_reliability/data0"
        np.save(filename,label_reliability)
        #with open(filename, mode='w') as file:  # 書き込み
        #    file.write(str(label_reliability))
        #    file.close()
        return label_reliability

        # ここまでadaboast


if __name__ == '__main__':
    mainmethod=Main()
    mainmethod.main()