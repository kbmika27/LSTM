#coding:utf-8
#エンコーダクラス
import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):
        super(Encoder,self).__init__()
        self.rnn=nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first = True)

    def forward(self, inputs, hidden0=None): #予測をする
        output,(state, cell) = self.rnn(inputs, hidden0)  # LSTM層　隠れ状態のベクトル
        return output,(state,cell)