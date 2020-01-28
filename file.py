#coding:utf-8
#読み込んで書き込む

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import SGD

#データセットを作る　入力は3*n,出力は3*n
def main():
    # ファイルの書き込み
    j = 0  # カウント用
    text = "tomtxt/tom-4.txt" #ここから
    f = open(text)  # ファイルの読み込み
    alldata = f.read()  # 全部読み込む
    f.close()
    linenum=int(sum(1 for line in open(text))) #行数カウント
    print("行数"+str(linenum))
    a = np.array([[1.0] * 2] * linenum)
    lines = alldata.split('\n')  # 改行で区切る
    for line in lines:  # 1行
        linedata = line.split(',')
        line_x = linedata[0]  # 各行のx座標 str
        line_y = linedata[1]  # 各行のy座標
        a[j][0] = float(line_x)
        a[j][1] = float(line_y)
        j += 1
    startnum=288 #スタートしたい番号-1
    count=startnum
    interval_num=4 #何個おきか
    overlap_num=5 #データの中身が何個おきか
    for i in range(1,linenum-19*overlap_num, interval_num):
        count+=1
        number=1
        filename = "test/xy_%d.txt" % count
        print(filename)
        with open(filename, mode='w') as file:  # 書き込み
            for j in range(20):
                print(i+j*overlap_num)
                x = a[i+j*overlap_num][0] #3個おき
                y = a[i+j*overlap_num][1]
                if (j < 19):
                    file.write(str(x) + ',')
                    file.write(str(y) + '\n')
                else:
                    file.write(str(x) + ',')
                    file.write(str(y))


if __name__ == '__main__':
    main()