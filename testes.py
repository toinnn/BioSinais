import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM , GPT2Tokenizer
import json
import matplotlib.pyplot as plt
import numpy as np


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

aux = json.load(open("CLS_token.json", "r"))
print(torch.tensor( json.load(open("CLS_token.json", "r")) ))

a = (np.array([1,2,3,4]) , np.array([1,2,3,4]) + 2)

plt.plot( range(1 , len(a[0]) + 1 ) , a[0])
plt.plot( range(1 , len(a[0]) + 1 ) , a[1])

plt.show()