#coding:utf-8
#LSTM　テスト用
import numpy as np
import math
from  keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def main():
    numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt'))  # 13
    LSTMlist_X=[]
    LSTMlist_Y=[] #正解データ保存list
    for s in range(12,13,1):
        text="gardentxt/mika%d.txt"%(s)
        test_a = np.array([[1.0] * 2] * numline)
        f = open(text)  # ファイルを開く
        alldata = f.read()  # xy_i.txtを全部読み込む
        scaler = MinMaxScaler(feature_range=(0, 1))  # 正規化の準備
        f.close()
        lines = alldata.split('\n')  # 改行で区切る
        j = 0  # カウント用
        for line in lines:  #1行ずつ読み込む
            linedata = line.split(',')
            line_x = linedata[0]  # 各行のx座標
            line_y = linedata[1]  # 各行のy座標
            test_a[j][0] = float(line_x)
            test_a[j][1] = float(line_y)
            j += 1
        test_label = np.array(test_a[10:len(test_a) + 1])
        LSTMlist_Y.append(test_label)
        test_a = scaler.fit_transform(test_a)
        test_a = np.array(test_a.tolist())
        test_input = []  # モデルへの入力に使うlist
        for i in range(10):
            test_in = test_a[i:i + 10]
            test_input.append(test_in)
        test_input = np.reshape(np.array(test_input), [10, 10, 2])
        LSTMlist_X.append(test_input)#入力データ

    model=load_model('modelstore/lstm_model', compile=False)
    ADE = 0
    FDE = 0.0
    for i in range(len(LSTMlist_X)):
        trainPredict = model.predict(LSTMlist_X[i])
        trainPredict = scaler.inverse_transform(trainPredict)
        # ADE
        ade_abstract = np.sqrt(np.square(np.abs(trainPredict - LSTMlist_Y[i])))
        ade_sum = (np.sum(ade_abstract)) / 10
        ADE+=ade_sum
        # FDE
        fde_abstract = np.abs(trainPredict[len(trainPredict) - 1] - LSTMlist_Y[len(LSTMlist_Y) - 1])
        fde_sum = math.sqrt(np.sum(np.square(fde_abstract)))
        FDE+=fde_sum
    print("ADE: " + str(ADE/len(LSTMlist_X)))
    print("FDE: " + str(FDE/len(LSTMlist_X)))

if __name__ == '__main__':
    main()