#coding:utf-8
import torch
import torch.nn as nn
#デコーダー
class Decoder(nn.Module):
    def __init__(self,hiddenDim,outputDim):  #初期化
        super(Decoder,self).__init__()
        self.bn1=nn.BatchNorm1d(outputDim) #バッチ正規化
        self.output_layer = nn.Linear(hiddenDim, outputDim)  # 全結合層

    def forward(self, concat): #予測をする
        output = self.output_layer(concat[:, -1, :])
        return output