#coding:utf-8
#attentionなしのLSTM 2次元　LSTM卒論に使う
import glob

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras import Sequential
from keras.layers import LSTM, Dense
from  keras.models import load_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# データセットを作る
def create_dataset(dataset, maxlen):
    # dataX = np.array([[[1.0] * 2] * maxlen] * (len(dataset)-maxlen-1)) #1つの番地に3つの値を持つ
    dataX = np.array([[[1.0 for i in range(2)] for j in range(maxlen)] for k in range(len(dataset) - maxlen - 1)])
    dataY = np.array([[1.0] * 2] * (len(dataset) - maxlen - 1))
    for i in range(len(dataset) - maxlen - 1):
        # a = np.array([[0] * 2] * maxlen)  # dataXの中の1つのセット aもdataX[i]も長さ3
        for j in range(maxlen):
            # a[j][0]=dataset[j][0]
            # a[j][1]=dataset[j][1]  aはいらないかも
            dataX[i][j][0] = dataset[i + j][0]
            dataX[i][j][1] = dataset[i + j][1]
        dataY[i] = dataset[i + maxlen]
    return np.array(dataX), np.array(dataY)

def main():
    #ここから書く
    filenum = glob.glob("/Users/kobayakawamika/PycharmProjects/LSTM/lstm_xy_data/*")  # ファイル数を取得する
    filenum = len(filenum)
    numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/lstm_xy_data/xy_0.txt'))  # 13
    trainX = []
    trainY = []
    for i in range(filenum):  # ファイルの読み込み
        j = 0  # カウント用
        a = np.array([[1.0] * 2] * numline)
        text = "lstm_xy_data/xy_%d.txt" % (i)
        f = open(text)  # ファイルを開く
        alldata = f.read()  # xy_i.txtを全部読み込む
        scaler = MinMaxScaler(feature_range=(0, 1))  # 正規化の準備
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
        train_in = []
        train_out = []
        for b in range(10):
            train_in.append(np.array(a[b:b+10]))
            train_out.append(np.array(a[b+10]))
        train_in=np.reshape(np.array(train_in),[2,10,10])
        train_out=np.reshape(np.array(train_out),[2,1,10])
        trainX.append(np.array(train_in))
        trainY.append(np.array(train_out))
    trainX = np.array(trainX) #50 10 10 2
    trainX=np.reshape(trainX,[500,10,2])
    trainY = np.array(trainY)
    trainY=np.reshape(trainY,[500,2])
    print(trainX.shape)
    print(trainY[:10, :])

    print("ここまで")
    #ここまで書く

    maxlen = 10  # t,t-1,t-2のデータセットを作成
    # lstmの学習モデルを作成する  ここにattentionを入れる
    model = Sequential()
    model.add(LSTM(4, input_shape=(maxlen, 2)))  # 4は隠れ層  input_shape=1,maxlenだったところを変更
    model.add(Dense(2))  # ニューロンの数を調節している
    print(trainX.shape)
    print(trainY.shape)
    model.compile(loss='mean_squared_error', optimizer='adam')  # 誤差関数、最適化法
    # 学習
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)  # epocが増えるとlossが減る batchでtrainingdataを分割
    model.save('lstm_model',include_optimizer=False)

if __name__ == '__main__':
    main()