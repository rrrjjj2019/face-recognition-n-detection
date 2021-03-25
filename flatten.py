import torch
import torch.nn as nn
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
 
    def forward(self,input):
        return input.view(input.size(0),-1)