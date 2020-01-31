#coding:utf-8
#attentionなしのLSTM 学習用
import glob
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# データセットを作る
def create_dataset(dataset, maxlen):
    dataX = np.array([[[1.0 for i in range(2)] for j in range(maxlen)] for k in range(len(dataset) - maxlen - 1)])
    dataY = np.array([[1.0] * 2] * (len(dataset) - maxlen - 1))
    for i in range(len(dataset) - maxlen - 1):
        for j in range(maxlen):
            dataX[i][j][0] = dataset[i + j][0]
            dataX[i][j][1] = dataset[i + j][1]
        dataY[i] = dataset[i + maxlen]
    return np.array(dataX), np.array(dataY)

def main():
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
            line_x=linedata[0] #各行のx座標 str type
            line_y=linedata[1] #各行のy座標
            a[j][0]=float(line_x)
            a[j][1]=float(line_y)
            j+=1
        a = scaler.fit_transform(a)
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
    maxlen = 10


    # lstmの学習モデルの作成
    model = Sequential()
    model.add(LSTM(4, input_shape=(maxlen, 2)))
    model.add(Dense(2))   #ニューロン数の調節
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    model.save('lstm_model',include_optimizer=False)  #モデルの保存

if __name__ == '__main__':
    main()