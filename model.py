import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):

    def __init__(self,model_type,ntoken,embedding_size,window_size,hidden_size,bptt,dropout=0.5, tie_weights=False):
        super(FNNModel, self).__init__()
        self.ntoken = ntoken
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.bptt = bptt
        #self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntoken, self.embedding_size)
        self.Middle = nn.Linear(self.window_size * self.embedding_size, self.hidden_size)
        self.act = torch.tanh
        self.decoder1 = nn.Linear(self.hidden_size,self.ntoken)
        #self.decoder2 = nn.Linear(self.window_size * self.embedding_size,self.ntoken)

        #print(tie_weights)
        if tie_weights:
            if self.embedding_size != self.hidden_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to embedding_size')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        #initrange = 0.1
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder1.weight)
        #nn.init.xavier_uniform_(self.decoder2.weight)
        nn.init.xavier_uniform_(self.Middle.weight)

    def forward(self, input):
        #emb = self.drop(self.encoder(input))
        #print(input.shape)
        emb = self.encoder(input)
        #newemb = newemb.permute(1,0,2)
        
        if self.window_size > 1:
            newemb = []
            firstsize = emb.shape[0]
            for i in range(0,firstsize - self.window_size + 1):
                oneemb = []
                for j in range(0,self.window_size):
                    #print(emb[:,i+j,:].shape)
                    oneemb.append(emb[i+j,:,:].unsqueeze(0))
                oneemb = torch.cat(oneemb,2)
                #print(oneemb.shape)
                newemb.append(oneemb)
            newemb = torch.cat(newemb,0)
            #print(newemb.shape)
        
        else:
            newemb = emb
        #print(newemb.shape)
        newemb = newemb.reshape(-1, self.window_size *self.embedding_size)
        #print(newemb.shape)
        output = self.act(self.Middle(newemb))
        #print(output.shape)
        #output = self.drop(output)
        decoded_1 = self.decoder1(output)
        #decoded_2 = self.decoder2(newemb)
        #print(decoded.shape)
        #print("---------------------------")
        return F.log_softmax(decoded_1, dim=1)
        #return F.log_softmax(decoded_1 + decoded_2, dim=1)