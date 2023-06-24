import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
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

    def weights(self)->list:
        weights = [self.linear_Out_reg] + [self.linear_Out_classes]
        for i in self.layers :
            weights += i.weights()
        return weights
    
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
        return soft_Out , reg__Out #torch.cat( (soft_Out , reg__Out) ,dim = 1)
    
    def forward(self ,Enc_values , Enc_keys , max_lengh = 100 ) :
        sequence = self.BOS
        classes_list = []#Indixe dos outputs das classes 
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
            
            classes_list.append(out_class)
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
        return classes_list , pos_list

class Ner_Class_And_Reg(nn.Module):
    def __init__(self,model_dim ,heads_Enc , heads_Dec ,num_Enc_layers ,num_Dec_layers  , num_Classes  ):
        super(Ner_Class_And_Reg,self).__init__()
        self.model_dim = model_dim
        # self.Embedding = Embedding
        self.encoder = encoder(model_dim , heads_Enc , num_Enc_layers)
        self.decoder = Ner_Class_And_Reg_Decoder(model_dim , heads_Dec , num_Dec_layers , num_Classes  )

    def sparsefy(self , amount_ = 0.2)->None:
        """Pruning the network """
        parameters = self.encoder.weights() + self.decoder.weights()
        parameters = [ (i , "weight") for i in parameters ]
        prune.global_unstructured( parameters , pruning_method=prune.L1Unstructured, amount = amount_)

    def fit(self ,batch_Input , batch_Output , maxAge , maxErro,n = 0.05 ,Betas = (0.9,.999) ,
            lossFunction_Clas = nn.CrossEntropyLoss() , lossFunction_Reg = nn.MSELoss() ,
            lossGraphNumber = 1 ):
        self.optimizer = torch.optim.Adam(self.parameters(), n ,Betas)
        lossValue = float("inf")
        Age = 0
        lossList_Cla = []
        lossList_Reg = []
        # batch_Input  = [ self.Embedding.sequence2vectors(i) for i in batch_Input  ]
        # batch_Input  = [ self.Embedding.sequence2vectors(i) for i in batch_Input  ]
        # batch_Output = [ self.Embedding.sequence2idx(i)     for i in batch_Output ]
        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            
            for x,y in zip(batch_Input,batch_Output) :
                print("y.shape[0] = {}".format(y.shape[0]))
                if type(y[0]) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = (torch.from_numpy(y[0]).float() , torch.from_numpy(y[1]).float())
                # div = len(y)
                enc = self.encoder(x , mask = False ,scale = True )
                print('____________DECODER ___________________\n\n\n\n')
                out_clas , out_reg = self.decoder.forward_fit(enc , enc , max_lengh = y[0].shape[0])
                
                
                
                loss_Cla = lossFunction_Clas(out_clas , y[0] )
                loss_Reg = lossFunction_Reg( out_reg  , y[1] )
                
                lossValue_Cla += loss_Cla.item()
                lossValue_Reg += loss_Reg.item()
                
                loss_Cla.backward()
                loss_Reg.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            Age += 1
            # lossValue = lossValue/div
            lossList_Cla.append(lossValue_Cla)
            lossList_Reg.append(lossValue_Reg)
        plt.plot(range(1 , Age + 1) , lossList_Cla)
        plt.plot(range(1 , Age + 1) , lossList_Reg)
        if lossGraphNumber != 1 :
            plt.savefig("{}_Tener_Cla_&_Reg__LossInTrain_Plot.png".format(lossGraphNumber) )
            # plt.savefig("{}_Tener_Cla_&_Reg__LossInTrain_Plot.pdf".format(lossGraphNumber) )
        else :
            plt.savefig("Tener_Cla_&_Reg__LossInTrain_Plot.png")
            # plt.savefig("Tener_Cla_&_Reg__LossInTrain_Plot.pdf")
        
        print("O erro final foi de {} ".format(lossValue))


    def forward(self , x ,Enc_mask = False ,Enc_scale = True ,max_lengh = 100 ) :
        enc = self.encoder(x , mask = Enc_mask ,scale = Enc_scale )
        out_class, out_pos = self.decoder(enc , enc , max_lengh = max_lengh )
        return out_class, out_pos


#Este modelo consiste em um grupo de regressores/(classificadores Bin) Sigmóide  :

class Ner_Class_Decoder(nn.Module):
    def __init__(self,model_dim ,heads ,num_layers  , num_Classes  ,forward_expansion = 4  ):
        super(Ner_Class_Decoder ,self).__init__()
        self.BOS = CLS_Embbeding.view(1,-1)
        self.EOS = SEP_Embbeding.view(1,-1)
        self.num_Classes = num_Classes

        self.layers = nn.ModuleList( decoderBlock(model_dim , heads , forward_expansion = forward_expansion) for _ in torch.arange(num_layers))
        
        self.linear_Out = nn.Linear(model_dim , num_Classes )
        
        self.out_sig = nn.Sigmoid()

    def weights(self)->list:
        weights = [self.linear_Out] 
        for i in self.layers :
            weights += i.weights()
        return weights
    
    def forward_fit(self ,Enc_values , Enc_keys ) :#, max_lengh  ) :
        sequence = self.BOS
        # soft_Out = [] # nn.ModuleList([])
        # reg__Out = []
        # sig__Out = []
        
        buffer = sequence
        for l in self.layers :
            buffer = l(buffer , Enc_values , Enc_keys)
        return self.out_sig(self.linear_Out_reg(buffer[-1])  )
    
        # return buffer
        # while  sequence.shape[0]<= max_lengh  :# Ta errado
        #     print("Mais um loop de Decoder e sequence.shape[0] = " , sequence.shape[0] )
        #     buffer = sequence
        #     for l in self.layers :
        #         buffer = l(buffer , Enc_values , Enc_keys)
        #     buffer_sig = self.out_sig(self.linear_Out_reg(buffer[-1])  )
            
        #     # out        = torch.argmax(buffer_class).item()
        #     # out = heapq.nlargest(1, enumerate(buffer ) , key = lambda x : x[1])[0]
        #     # soft_Out.append(buffer_class.view(1,-1)*(1/self.num_Classes) )
        #     sig__Out.append(buffer_reg.view(1,-1)  *(1/self.num_Classes) )
            
        #     # sequence = torch.cat((sequence , torch.from_numpy(self.embedding[ self.embedding.index2word[ out ] ] ).float().view(1,-1)),dim = 0 )
        #     # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
        #     sequence = torch.cat((sequence , self.BOS ) , dim = 0 )

        # # soft_Out = torch.cat(soft_Out ,dim = 0)
        # # sig__Out = torch.cat(sig__Out ,dim = 0)
        # return torch.cat(sig__Out ,dim = 0)
    
    def forward(self ,Enc_values , Enc_keys, masked = True) : # , max_lengh = 100 , masked = True) :
        sequence = self.BOS
        # idx = []#Indixe dos outputs das classes 
        # pos_list = []# [tensor([[Begin of Entity , End of Entity]])]
        
        buffer = sequence
        for l in layers :
            buffer = l(buffer , Enc_values , Enc_keys)
        buffer = self.out_sig(self.linear_Out_reg(buffer[-1]) )
        if masked :
            return buffer.ge(.5)
        else:
            return buffer
        # while sequence[-1] != self.EOS and sequence.shape[0]< max_lengh  :
        #     buffer = sequence
        #     for l in layers :
        #         buffer = l(buffer , Enc_values , Enc_keys)
        #     # buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
        #     buffer_sig = self.out_sig(self.linear_Out_reg(buffer[-1]) )
        #     # out_class , out_pos       = torch.argmax(buffer_class).item() , buffer_reg
        #     # out = heapq.nlargest(1, enumerate(buffer_class ) , key = lambda y : y[1])[0]
            
        #     idx.append(out_class)
        #     pos_list.append(out_pos.view(-1) )
        #     # idx.append(out[0])
        #     #buffer = F.softmax(buffer , dim = 1)
        #     #buffer = O Vetor com a maior probabilidade , mas qual ??
            
        #     if out_class == self.num_Classes :
        #         sequence = torch.cat((sequence , self.EOS  ),dim = 0 )
        #     else :
        #         sequence = torch.cat((sequence , self.BOS  ),dim = 0 )
        
        # # sequence = [self.embedding.idx2token[i] for i in idx ]
        # # return sequence
        # return idx , pos_list

class Ner_Class(nn.Module):
    def __init__(self,model_dim ,heads_Enc , heads_Dec ,num_Enc_layers ,num_Dec_layers  , num_Classes  ):
        super(Ner_Class ,self).__init__()
        self.model_dim = model_dim
        self.num_classes = num_Classes 
        # self.Embedding = Embedding
        self.encoder = encoder(model_dim , heads_Enc , num_Enc_layers)
        self.decoder = Ner_Class_Decoder(model_dim , heads_Dec , num_Dec_layers , num_Classes  )

    def sparsefy(self , amount_ = 0.2)->None:
        """Pruning the network """
        parameters = self.encoder.weights() + self.decoder.weights()
        parameters = [ (i , "weight") for i in parameters ]
        prune.global_unstructured( parameters , pruning_method=prune.L1Unstructured, amount = amount_)
    
    def fit_step(self , x , y , lossFunction = nn.CrossEntropyLoss()):

        enc = self.encoder(x , mask = False ,scale = True )
        print('____________DECODER ___________________\n\n\n\n')
        out = self.decoder.forward_fit(enc , enc )
        
        # print("out.shape = " , out.shape ,"\nmult_oneHotEncode(self.model_dim, y ).shape = " , mult_oneHotEncode(len(self.Embedding.vocab), y ).shape )
        # loss = lossFunction(out , mult_oneHotEncode(len(self.Embedding.vocab), y ))
        loss = lossFunction(out , y )
        lossValue = loss.item()
        loss.backward()

        return lossValue


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

                if x.shape[1] > self.num_classes :
                    splits = x.shape[1]/self.num_Classes
                    
                    x_buffer  = [ x[0][i*self.num_Classes : (i+1)*self.num_Classes].view(1,-1 )  for i in range(0, splits )]
                    last_x = [x[0][(i+1)*self.num_Classes :].view(1,-1 )]
                    last_x = [torch.cat(last_x + [torch.zeros(1 , x_buffer[0].shape[1] - last_x[0].shape[1]  ) ] , dim = 1)]
                    x = x_buffer + last_x #torch.cat( x_buffer + last_x , dim= 2) 

                    y_buffer = [ y[0][i *self.num_Classes  : (i+1) * self.num_Classes ].view(1,-1)  for i in range(0, splits )]
                    last_y = [y[0][(i+1)*self.num_Classes  : ].view(1,-1)]
                    last_y = [torch.cat( last_y + [torch.zeros( 1 , y_buffer[0].shape[1] - last_y[0].shape[1]  ) ] , dim = 1)]
                    y = y_buffer + last_y #torch.cat( y_buffer + last_y , dim = 2 )

                    for i,j in zip(x,y):
                        lossValue += self.fit_step(i , j , lossFunction )


                lossValue += self.fit_step(x , y , lossFunction )
                
                # div = len(y)
                # enc = self.encoder(x , mask = False ,scale = True )
                # print('____________DECODER ___________________\n\n\n\n')
                # out = self.decoder.forward_fit(enc , enc )
                
                # # print("out.shape = " , out.shape ,"\nmult_oneHotEncode(self.model_dim, y ).shape = " , mult_oneHotEncode(len(self.Embedding.vocab), y ).shape )
                # # loss = lossFunction(out , mult_oneHotEncode(len(self.Embedding.vocab), y ))
                # loss = lossFunction(out , y )
                # lossValue += loss.item()
                # loss.backward()
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

    def forward(self , x ,Enc_mask = False ,Enc_scale = True , masked_out = True ) : # ,max_lengh = 100 ) :
        enc = self.encoder(x , mask = Enc_mask ,scale = Enc_scale )
        out_class = self.decoder(enc , enc , masked = masked_out )
        return out_class
    
class Ner_Class_Handler(nn.Module):
    def __init__(self,model_dim ,heads_Enc , heads_Dec ,num_Enc_layers ,num_Dec_layers  , num_Classes  ):
        super(Ner_Class_Handler ,self).__init__()
        self.Ner = Ner_Class(model_dim ,heads_Enc , heads_Dec ,num_Enc_layers ,num_Dec_layers  , num_Classes)
        self.num_Classes = num_Classes
        self.model_dim = model_dim

    def sparsefy(self , amount__ = 0.2)->None:
        """Pruning the network """
        self.Ner.sparsefy(amount_= amount__) 
        
    def fit(self ,batch_Input , batch_Output , maxAge , maxErro,n = 0.05 ,Betas = (0.9,.999) ,  lossFunction = nn.CrossEntropyLoss() , 
            lossGraphNumber = 1):
        # for x,y in zip(batch_Input , batch_Output):
        # if x.shape[0]> self.num_Classes :
        #     for i in range(0 , )
        # self.Ner.fit(batch_Input , batch_Output , maxAge , maxErro,n ,Betas  ,  lossFunction , 
        #     lossGraphNumber)
        # return
        self.Ner.fit(batch_Input , batch_Output , maxAge , maxErro,n  ,Betas  ,  lossFunction  , 
            lossGraphNumber)

    def forward(self , x ,Enc_mask = False ,Enc_scale = True , masked_out = True ) :

        if x.shape[0]> self.num_Classes :
            splits = x.shape[0]/self.num_Classes
            if type(splits) == type(x.shape[0]/1) :
                return torch.cat(( self.Ner(x[i*self.num_Classes : (i+1)*self.num_Classes ] , masked = masked_out) for i in range(0, splits )) , dim=0)
            buffer = [ self.Ner(x[i*self.num_Classes : (i+1)*self.num_Classes] , masked = masked_out ) for i in range(0, splits )]
            return torch.cat(buffer + [self.Ner(x[ int(splits)*self.num_Classes : ] , masked = masked_out )] , dim = 0)
        return self.Ner(x ,Enc_mask = False ,Enc_scale = True  , masked = masked_out)
