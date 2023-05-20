import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import random as rd
from torch.autograd import Variable

from Tener import decoderBlock , encoder
import json

CLS_Embbeding = torch.tensor( json.load(open("CLS_token.json", "r")) )
SEP_Embbeding = torch.tensor( json.load(open("SEP_token.json", "r")) )
print("Rodei MUUUito bem")

#Este modelo consiste em 2 regressores Sigmóide e 1 classificador Softmax como OutPut :
class Ner_Class_And_Reg_Decoder(nn.Module):
    def __init__(self,model_dim ,heads ,num_layers  , num_Classes  ,forward_expansion = 4  ):
        super(Ner_Class_And_Reg_Decoder,self).__init__()
        self.BOS = CLS_Embbeding.view(1,-1)
        self.EOS = SEP_Embbeding.view(1,-1)
        self.num_Classes = num_Classes

        self.layers = nn.ModuleList( decoderBlock(model_dim , heads , forward_expansion = forward_expansion) for _ in torch.arange(num_layers))
        
        self.linear_Out_classes = nn.Linear(model_dim , num_Classes )
        self.linear_Out_reg = nn.Linear(model_dim ,  2)
        self.out_sig = nn.Sigmoid()


    def forward_fit(self ,Enc_values , Enc_keys , max_lengh  ) :
        sequence = self.BOS
        soft_Out = [] # nn.ModuleList([])
        reg__Out = []
        

        while  sequence.shape[0]<= max_lengh  :# Ta errado
            print("Mais um loop de Decoder e sequence.shape[0] = " , sequence.shape[0] )
            buffer = sequence
            for l in self.layers :
                buffer = l(buffer , Enc_values , Enc_keys)
            buffer_class , buffer_reg = F.softmax(self.linear_Out_classes(buffer[-1]) , dim = 0 ) ,
            self.out_sig(self.linear_Out_reg(buffer[-1])  )
            
            # out        = torch.argmax(buffer_class).item()
            # out = heapq.nlargest(1, enumerate(buffer ) , key = lambda x : x[1])[0]
            soft_Out.append(buffer_class.view(1,-1)*(1/self.num_Classes) )
            reg__Out.append(buffer_reg.view(1,-1)  *(1/2) )
            
            # sequence = torch.cat((sequence , torch.from_numpy(self.embedding[ self.embedding.index2word[ out ] ] ).float().view(1,-1)),dim = 0 )
            # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
            sequence = torch.cat((sequence , self.BOS ) , dim = 0 )

        soft_Out = torch.cat(soft_Out ,dim = 0)
        reg__Out = torch.cat(reg__Out ,dim = 0)
        return torch.cat( (soft_Out , reg__Out) ,dim = 1)
    
    def forward(self ,Enc_values , Enc_keys , max_lengh = 100 ) :
        sequence = self.BOS
        idx = []#Indixe dos outputs das classes 
        pos_list = []# [tensor([[Begin of Entity , End of Entity]])]
        while sequence[-1] != self.EOS and sequence.shape[0]< max_lengh  :
            buffer = sequence
            for l in layers :
                buffer = l(buffer , Enc_values , Enc_keys)
            # buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            buffer_class , buffer_reg = F.softmax(self.linear_Out_classes(buffer[-1]) , dim = 0 ) ,
            self.out_sig(self.linear_Out_reg(buffer[-1]) )
            out_class , out_pos       = torch.argmax(buffer_class).item() , buffer_reg
            # out = heapq.nlargest(1, enumerate(buffer_class ) , key = lambda y : y[1])[0]
            
            idx.append(out_class)
            pos_list.append(out_pos.view(-1) )
            # idx.append(out[0])
            #buffer = F.softmax(buffer , dim = 1)
            #buffer = O Vetor com a maior probabilidade , mas qual ??
            
            if out_class == self.num_Classes :
                sequence = torch.cat((sequence , self.EOS  ),dim = 0 )
            else :
                sequence = torch.cat((sequence , self.BOS  ),dim = 0 )
        
        # sequence = [self.embedding.idx2token[i] for i in idx ]
        # return sequence
        return idx , pos_list

class Ner_Class_And_Reg(nn.Module):
    def __init__(self,model_dim ,heads_Enc , heads_Dec ,num_Enc_layers ,num_Dec_layers  , num_Classes  ):
        super(Ner_Class_And_Reg,self).__init__()
        self.model_dim = model_dim
        self.Embedding = Embedding
        self.encoder = encoder(model_dim , heads_Enc , num_Enc_layers)
        self.decoder = Ner_Class_And_Reg_Decoder(model_dim , heads_Dec , num_Dec_layers , num_Classes  )

    def fit(self ,batch_Input , batch_Output , maxAge , maxErro,n = 0.05 ,Betas = (0.9,.999) ,  lossFunction = nn.CrossEntropyLoss() , 
            lossGraphNumber = 1 ):
        self.optimizer = torch.optim.Adam(self.parameters(), n ,Betas)
        lossValue = float("inf")
        Age = 0
        lossList = []

        # batch_Input  = [ self.Embedding.sequence2vectors(i) for i in batch_Input  ]
        # batch_Input  = [ self.Embedding.sequence2vectors(i) for i in batch_Input  ]
        # batch_Output = [ self.Embedding.sequence2idx(i)     for i in batch_Output ]
        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            
            for x,y in zip(batch_Input,batch_Output) :
                print("y.shape[0] = {}".format(y.shape[0]))
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                # div = len(y)
                enc = self.encoder(x , mask = False ,scale = True )
                print('____________DECODER ___________________\n\n\n\n')
                out = self.decoder.forward_fit(enc , enc , max_lengh = y.shape[0])
                
                # print("out.shape = " , out.shape ,"\nmult_oneHotEncode(self.model_dim, y ).shape = " , mult_oneHotEncode(len(self.Embedding.vocab), y ).shape )
                # loss = lossFunction(out , mult_oneHotEncode(len(self.Embedding.vocab), y ))
                loss = lossFunction(out , y )
                lossValue += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            Age += 1
            # lossValue = lossValue/div
            lossList.append(lossValue)
        plt.plot(range(1 , Age + 1) , lossList)
        if lossGraphNumber != 1 :
            plt.savefig("{}_Tener_LossInTrain_Plot.png".format(lossGraphNumber) )
            plt.savefig("{}_Tener_LossInTrain_Plot.pdf".format(lossGraphNumber) )
        else :
            plt.savefig("Tener_LossInTrain_Plot.png")
            plt.savefig("Tener_LossInTrain_Plot.pdf")
        
        print("O erro final foi de {} ".format(lossValue))


    def forward(self , x ,Enc_mask = False ,Enc_scale = True ,max_lengh = 100 ) :
        enc = self.encoder(x , mask = Enc_mask ,scale = Enc_scale )
        out_class, out_pos = self.decoder(enc , enc , max_lengh = max_lengh )
        return out_class, out_pos
#Este modelo consiste em 2 regressores Sigmóide  :