#coding:utf-8
#ファイルの読み込みと生成

import numpy as np

def main():
    # ファイルの書き込み
    for i in range(1,20,1):
        j = 0  # カウント用
        a = np.array([[1.0] * 2] * 20)
        number = i
        readtext = "xy0_data/xy_10.txt"
        f = open(readtext)  # readtextを読み込む
        alldata = f.read()
        f.close()
        lines = alldata.split('\n')  # 改行で区切る
        for line in lines:  # 1行
            linedata = line.split(',')
            line_x = linedata[0]  # 各行のx座標
            line_y = linedata[1]  # 各行のy座標
            a[j][0] = float(line_x)
            a[j][1] = float(line_y)
            j += 1
        #filename = "test/xy_%d.txt" % number
        filename = "test/xy_0.txt"
        with open(filename, mode='w') as file:  # 書き込み
            for k in range(20):
                if(k<19):
                    file.write(str(a[k][0])+ ','+str(a[k][1]) + '\n')
                else:
                    file.write(str(a[k][0]) + ','+str(a[k][1]))
if __name__ == '__main__':
    main()