import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):

    def __init__(self,model_type,ntoken,embedding_size,window_size,hidden_size,dropout=0.5, tie_weights=False):
        super(FNNModel, self).__init__()
        self.ntoken = ntoken
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntoken, self.embedding_size)
        self.Middle = nn.Linear(self.window_size * self.embedding_size, self.hidden_size)
        self.act = torch.tanh
        self.decoder1 = nn.Linear(self.hidden_size,self.ntoken)
        #self.decoder2 = nn.Linear(self.window_size * self.embedding_size,self.ntoken)

        print(tie_weights)
        if tie_weights:
            if self.embedding_size != self.hidden_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to embedding_size')
            self.decoder1.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        #initrange = 0.1
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder1.weight)
        #nn.init.xavier_uniform_(self.decoder2.weight)
        nn.init.xavier_uniform_(self.Middle.weight)

    def getembedding(self,input):
        emb = self.encoder(input)
        return emb

    def forward(self, input):
        #emb = self.drop(self.encoder(input))
        #print(input.shape)
        emb = self.encoder(input).view(-1, self.window_size * self.embedding_size)
        #emb = emb.view(-1, self.window_size * self.embedding_size)
        output = self.act(self.Middle(emb))
        #print(output.shape)
        output = self.drop(output)
        decoded_1 = self.decoder1(output)
        #print(decoded_1.shape)
        #decoded_2 = self.decoder2(emb)
        #print(decoded_2.shape)
        #print(decoded.shape)
        return F.log_softmax(decoded_1, dim=1)
        #return F.log_softmax(decoded_1 + decoded_2, dim=1)
