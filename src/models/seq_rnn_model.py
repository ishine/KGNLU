#!/usr/bin/env python3
#coding=utf8
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence 

class SeqRNNModel(nn.Module):

    def __init__(self,namespace,voc_size,slot_tagset_size,sentence_tagset_size):
        super(SeqRNNModel,self).__init__()
        self._parse_namespace(namespace)
        self.vocabulary_size=voc_size # plus padding
        self.slot_tagset_size=slot_tagset_size
        self.sentence_tagset_size=sentence_tagset_size
        
        self.word_embeddings=nn.Embedding(
            num_embeddings=self.vocabulary_size+1, #plus 1 due to padding_idx
            embedding_dim=self.embedding_size,
            padding_idx=self.vocabulary_size # use padding_idx to denote padding chars
        )
        self.input_drop_layer=nn.Dropout(p=self.input_dropout)
        self.rnn=getattr(nn,self.cell)(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirection,
        ) 
        self.num_directions=2 if self.bidirection else 1
        self.hidden2slot=nn.Linear(self.hidden_size*self.num_directions,slot_tagset_size)
        self.hidden2class=nn.Linear(self.hidden_size*self.num_directions,sentence_tagset_size)

    def init_hidden_states(self,input_data):
        # return (h0,c0)
        batch_size=input_data.size(0) if self.rnn.batch_first else input_data.size(1)
        h0=Variable(torch.zeros(self.layers*self.num_directions,batch_size,self.hidden_size),requires_grad=False)
        c0=Variable(torch.zeros(self.layers*self.num_directions,batch_size,self.hidden_size),requires_grad=False)
        if self.cuda_:
            h0=h0.cuda()
            c0=c0.cuda()
        return (h0,c0)

    def _parse_namespace(self,namespace):
        self.input_dropout=namespace.input_dropout
        self.dropout=namespace.dropout
        self.bidirection=namespace.bidirection
        self.layers=namespace.layers
        self.cell=namespace.cell
        self.use_pretrained_embeddings=namespace.use_word_embeddings
        self.embedding_size=namespace.embedding_size
        self.hidden_size=namespace.hidden_size
        self.cuda_=namespace.cuda

    def forward(self,input_data,input_length):
        if self.cell=='LSTM':
            h0_c0=self.init_hidden_states(input_data)
        else:
            h0,_=self.init_hidden_states(input_data)
        embeds=self.word_embeddings(input_data)
        embeds=self.input_drop_layer(embeds)
        # print(embeds.size())
        # pad variable sequence
        packed_embeds=pack_padded_sequence(embeds,input_length,batch_first=True)
        if self.cell=='LSTM':
            rnn_layer_out,(ht,ct)=self.rnn(packed_embeds,h0_c0)
        else:
            rnn_layer_out,ht=self.rnn(packed_embeds,h0)
        # rnn_layer_out size(): batch_size, sequence_length, hidden_size*bidirection
        # ht_ct size(): ((layers*bidirecetion,batch_size,hidden_size),(layers*bidirecetion,batch_size,hidden_size))
        
        unpacked_rnn_layer_out,_=pad_packed_sequence(rnn_layer_out,batch_first=True)
        # print(unpacked_rnn_layer_out.size())
        slot_out=F.log_softmax(self.hidden2slot(unpacked_rnn_layer_out),dim=2)
        # print(slot_out.size())

        # sentence classification
        index_slices=[self.layers*2-2,self.layers*2-1] if self.bidirection else [self.layers*1-1]
        index_slices=Variable(torch.LongTensor(index_slices))
        if self.cuda_:
            index_slices=index_slices.cuda()

        ht=torch.index_select(ht,0,index_slices)
        ht_t=ht.transpose(0,1) #change the order
        ht_reshape=ht_t.contiguous().view(ht_t.size(0),-1)
        sentence_out=F.log_softmax(self.hidden2class(ht_reshape),dim=1)
        return slot_out,sentence_out

    def load_module(self,load_path):
        self.load_state_dict(torch.load(open(load_path,'rb')))

    def save_module(self,save_path_name):
        torch.save(self.state_dict(),open(save_path_name,'wb'))
    