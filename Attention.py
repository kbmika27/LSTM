#coding:utf-8
import torch
import torch.nn as nn
import numpy as np
import math
#アテンション
class Attention(nn.Module):
    def __init__(self, inputDim, hiddenDim):  # 初期化
        super(Attention, self).__init__()
        self.hiddenDim = hiddenDim
        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True)

    def forward(self, inputs,hidden0,encoder_output):
        att_output, (hidden, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル ここには全部コピーを入れる
        train_in=10
        train_out=10
        weight = np.array([[1.0] * train_in] * train_out)  # 3*10 an(n)を格納
        for l in range(train_out):  # 3
            k_t_sum = 0  # 正規化するため分子を足す
            for k in range(train_in):  # 10
                t = att_output[0][l].t()  # デコーダーの転置
                k_t = torch.dot(t, encoder_output[0][k])  # 内積
                k_t = math.exp(k_t.data.item())
                weight[l][k] = k_t
                k_t_sum += k_t
            for w in range(train_in):  # 正規化
                weight[l][k] = weight[l][k] / k_t_sum
        cn = []  # 4*1が3つ格納されるはず c0 c1 c2
        for l in range(train_out):  # 3
            cn_sum = 0
            for k in range(train_in):  # 10
                for h in range(4):
                    encoder_output[0][k][h] = weight[l][k] * encoder_output[0][k][h]
                cn_sum += encoder_output[0][k]  # m=0~10のシグマ
            cn.append(cn_sum)  # .data付けても付けなくてもOK!
        #print("cn"+str(cn))
        concatlist=[]
        for c in range(train_out):
            concat=torch.cat([cn[c], att_output[0][c]], dim=0) #8*1のtensor型
            concatlist.append(concat.detach().numpy())
        concatlist=torch.tensor([concatlist]).float()
        return concatlist