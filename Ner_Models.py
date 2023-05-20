import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import random as rd
from torch.autograd import Variable

from Tener import decoderBlock
import json

CLS_Embbeding = torch.tensor( json.load(open("CLS_token.json", "r")) )
SEP_Embbeding = torch.tensor( json.load(open("SEP_token.json", "r")) )
print("Rodei MUUUito bem")

#Este modelo consiste em 2 regressores Sigmóide e 1 classificador Softmax como OutPut :
class Ner_Class_And_Reg(nn.Module):
    def __init__(self,model_dim ,heads ,num_layers ,word_Embedding ,EOS , num_Classes  ,forward_expansion = 4):
        super(decoder,self).__init__()
        self.BOS = CLS_Embbeding
        self.EOS = SEP_Embbeding

        self.layers = nn.ModuleList( decoderBlock(model_dim , heads , forward_expansion = forward_expansion) for _ in torch.arange(num_layers))
        
        self.linear_Out_classes = nn.Linear(model_dim , num_Classes )
        self.linear_Out_reg = nn.Linear(model_dim ,  2)

#Este modelo consiste em 2 regressores Sigmóide  :