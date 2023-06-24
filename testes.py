import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM , GPT2Tokenizer
import json
import matplotlib.pyplot as plt
import numpy as np
import heapq
from dataclasses import dataclass




# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# % matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#.wordpiece_tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# print(tokenizer.encode("[CLS]"))
# print(tokenizer(["[CLS]"] , ["[CLS]"]))

"""aux = json.load(open("CLS_token.json", "r"))
print(torch.tensor( json.load(open("CLS_token.json", "r")) ))

a = (np.array([1,2,3,4]) , np.array([1,2,3,4]) + 2)

plt.plot( range(1 , len(a[0]) + 1 ) , a[0])
plt.plot( range(1 , len(a[0]) + 1 ) , a[1])

plt.show() """


a   = torch.nn.Linear(4,5)
seq = torch.nn.Sequential(nn.Linear(4,5) , nn.Linear(5,4) , nn.Linear(4,5) )
# print(a(torch.tensor([1.0,1.0,1.0,1.0])))
# print(a.weight)
s = heapq.nsmallest( 10 , a.weight.view(-1)  , lambda x : abs(x.item()) )
b = [ i if i in s else 0 for i in a.weight.view(-1) ]  
# print( torch.tensor(b).view(5,4) )
# prune.random_unstructured(a, name="weight", amount=0.3)

prune.l1_unstructured(a, name="weight", amount=0.5 )
v1 = Variable(torch.rand(4 , 5 , dtype = float) , requires_grad = True).float()
v2 = Variable(torch.rand(5 , 4 , dtype = float) , requires_grad = True).float()

# @dataclass
# class Variable_wraple():
#     weight : Variable 
class Variable_wraple(nn.Module):
    def __init__(self , weight : Variable ) -> None:
        super(Variable_wraple , self ).__init__()
        self.weight = weight
    
v1_w = Variable_wraple(weight=v1)
v2_w = Variable_wraple(weight=v2)

parameters = list( (i , "weight") for i in seq)
parameters += [(v1_w , "weight") , (v2_w , "weight" ) ]
prune.global_unstructured( parameters , pruning_method=prune.L1Unstructured, amount=0.5  )
# print(a.weight)

# print(parameters)
# print(list(i.weight for i in seq))
print(v1)



# print(a(torch.tensor([1.0,1.0,1.0,1.0])))
