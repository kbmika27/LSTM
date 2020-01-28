#coding:utf-8
#attentionなしのLSTM 2次元　LSTMのテスト用
import glob
import numpy as np
import math
from  keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def main():
    #ここからテスト
    numline = sum(1 for line in open('/Users/kobayakawamika/PycharmProjects/LSTM/xy0_data/xy_0.txt'))  # 13
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
    test_label=np.array(test_a[10:len(test_a)+1]) #10 2
    test_a = scaler.fit_transform(test_a)  # これを正規化するので合ってるのか？
    test_a = np.array(test_a.tolist())
    print(test_a) #lan 20
    test_input=[] #モデルへの入力に使う
    for i in range(10):
        test_in=test_a[i:i+10]
        test_input.append(test_in)
    test_input=np.reshape(np.array(test_input),[10,10,2])

    model=load_model('lstm_model', compile=False)
    trainPredict = model.predict(test_input) #10 10 2を入力
    trainPredict = scaler.inverse_transform(trainPredict) #10 2
    # 未来の予測
    # ADEの計算 2嬢　ルート　たす　わる
    ade_abstract = np.sqrt(np.square(np.abs(trainPredict - test_label)))
    ade_sum = (np.sum(ade_abstract)) / 10
    print("ADE: " + str(ade_sum))

    # FDE
    fde_abstract = np.abs(trainPredict[len(trainPredict) - 1] - test_label[len(test_label) - 1])
    fde_sum = math.sqrt(np.sum(np.square(fde_abstract)))
    print("FDE: " + str(fde_sum))


if __name__ == '__main__':
    main()