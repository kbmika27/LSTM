#coding:utf-8
#デコーダー
import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self,hiddenDim,outputDim):
        super(Decoder,self).__init__()
        self.bn1=nn.BatchNorm1d(outputDim)
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(self, concat):
        output = self.output_layer(concat[:, -1, :])
        return output