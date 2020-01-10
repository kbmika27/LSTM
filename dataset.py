#coding:utf-8
#ファイルの読み込みと生成
#xyのファイル書き込み
import numpy as np


def main():
    # ファイルの書き込み
    a = np.array([[1.0] * 2] * 20)
    j = 0  # カウント用
    text = "person0/xy_0.txt"
    f = open(text)  # ファイルの読み込み
    alldata = f.read()  # 全部読み込む
    f.close()
    lines = alldata.split('\n')  # 改行で区切る
    for line in lines:  # 1行
        linedata = line.split(',')
        line_x = linedata[0]  # 各行のx座標 str
        line_y = linedata[1]  # 各行のy座標
        a[j][0] = float(line_x)
        a[j][1] = float(line_y)
        j += 1
    print(a)
    for i in range(1, 200, 1):
        number = i
        filename = "person0/xy_%d.txt" % number
        with open(filename, mode='w') as file:  # 書き込み
            for j in range(20):
                x = a[j][0] + np.random.normal(loc=0.0, scale=2)  # 前のx
                y = a[j][1] + np.random.normal(loc=0.0, scale=2)  # 前のy
                if (j < 19):
                    file.write(str(x) + ',')
                    file.write(str(y) + '\n')
                else:
                    file.write(str(x) + ',')
                    file.write(str(y))


if __name__ == '__main__':
    main()