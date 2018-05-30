#!/usr/bin/env python3
#coding=utf8
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence 
from .loss_function import sentence_loss_function

def log_sum_exp(array):
    if array.dim()==2:
        # array:[tagset,tagset]
        # return: [tagset]
        max_value,_=torch.max(array,dim=1) # max_value,index:[tagset]
        max_value_broadcast=max_value.unsqueeze(dim=1)
        array=torch.exp(array-max_value_broadcast)
        array=torch.sum(array,dim=1) # array:[tagset]
        return torch.log(array)+max_value
    else:
        # array.dim()==1
        max_value=torch.max(array)
        return max_value+torch.log(torch.sum(torch.exp(array-max_value)))

class SeqRNNCRFModel(nn.Module):

    def __init__(self,namespace,voc_size,slot_tagset_size,sentence_tagset_size):
        super(SeqRNNCRFModel,self).__init__()
        self._parse_namespace(namespace)
        self.vocabulary_size=voc_size
        self.slot_tagset_size=slot_tagset_size+2 # plus 2 cauze <BOS> and <EOS> symbols
        self.START_TAG='<BOS>'
        self.END_TAG='<EOS>'
        self.tag_to_ix={self.START_TAG:self.slot_tagset_size-2,self.END_TAG:self.slot_tagset_size-1} 
        self.sentence_tagset_size=sentence_tagset_size
        
        self.word_embeddings=nn.Embedding(
            num_embeddings=self.vocabulary_size+1, #plus 1 due to padding_idx,其实也可以没有，只要传入length_list
            embedding_dim=self.embedding_size,
            padding_idx=self.vocabulary_size # use padding_idx to denote padding chars, word embeddings will be [0,0,0...]
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
        self.transitions = nn.Parameter(torch.randn(self.slot_tagset_size, self.slot_tagset_size))
        self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -1000000
        self.transitions.data[:, self.tag_to_ix[self.END_TAG]] = -1000000
        self.hidden2tag=nn.Linear(self.hidden_size*self.num_directions,self.slot_tagset_size)
        self.hidden2class=nn.Linear(self.hidden_size*self.num_directions,self.sentence_tagset_size) # 句子分类结果

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

    def _rnn_features(self,input_data,input_length):
        if self.cell=='LSTM':
            h0_c0=self.init_hidden_states(input_data)
        else:
            h0,_=self.init_hidden_states(input_data)
        embeds=self.word_embeddings(input_data)
        embeds=self.input_drop_layer(embeds)
        # pad variable sequence
        packed_embeds=pack_padded_sequence(embeds,input_length,batch_first=True)
        if self.cell=='LSTM':
            rnn_layer_out,(ht,ct)=self.rnn(packed_embeds,h0_c0) #注意h_t和c_t是[num_layers * num_directions, batch, hidden_size]
        else:
            rnn_layer_out,ht=self.rnn(packed_embeds,h0)
        # rnn_layer_out size(): batch_size, sequence_length, hidden_size*bidirection
        # ht_ct size(): ((layers*bidirecetion,batch_size,hidden_size),(layers*bidirecetion,batch_size,hidden_size))
        unpacked_rnn_layer_out,_=pad_packed_sequence(rnn_layer_out,batch_first=True)
        # return [batch_size,len(sequence),hidden_size*num_directions]
        slot_out=F.log_softmax(self.hidden2tag(unpacked_rnn_layer_out),dim=2)
        # sentence classification
        index_slices=[self.layers*2-2,self.layers*2-1] if self.bidirection else [self.layers*1-1] #选取双向的最后一个中间状态拼接
        index_slices=Variable(torch.LongTensor(index_slices))
        if self.cuda_:
            index_slices=index_slices.cuda()

        ht=torch.index_select(ht,0,index_slices)
        ht_t=ht.transpose(0,1) #change the order,batch first
        ht_reshape=ht_t.contiguous().view(ht_t.size(0),-1)
        sentence_out=F.log_softmax(self.hidden2class(ht_reshape),dim=1)
        return slot_out,sentence_out

    def cross_entropy_loss(self,input_data,input_length,slot_tags,sentence_tags):
        slot_features,sentence_results=self._rnn_features(input_data,input_length)
        # slot_features:[batch_size,len(sequence),len(slot_tagset)]
        # sentence_results:[batch_size,len(sentence_tag_size)]
        sentence_scores=sentence_loss_function(sentence_results,sentence_tags)
        # sentence classification loss
        forward_score=self._forward_alg(slot_features,input_length)
        # forward_score=self._forward_alg_fast(slot_features,input_length)
        godden_score=self._score_given_sentence(slot_features,slot_tags,input_length)
        # godden_score=self._score_given_sentence_fast(slot_features,slot_tags,input_length)
        final_score=forward_score-godden_score+sentence_scores
        return final_score

    def _forward_alg_fast(self,slot_features,input_length):
        pass

    def _forward_alg(self,slot_features,input_length):
        # slot_features:[batch_size,len(sequence),hidden_size]
        init_log_alpha=torch.ones(self.slot_tagset_size)
        init_log_alpha=init_log_alpha*-1000000.
        # START_TAG has all of the score
        init_log_alpha[self.tag_to_ix[self.START_TAG]]=0.
        score=Variable(torch.zeros(1))
        if self.cuda_:
            init_log_alpha=init_log_alpha.cuda()
            score=score.cuda()
        for i,feats in enumerate(slot_features):
            length=input_length[i]
            forward_var=Variable(init_log_alpha) # [tagset]
            for idx,feat in enumerate(feats): #feats:max_length*tagset, feat:tagset
                idx+=1
                emit_scores=feat.unsqueeze(dim=1) # feat: [tagset,1]
                trans_scores=self.transitions # [tagset,tagset]
                forward_var=log_sum_exp(forward_var+emit_scores+trans_scores) # broadcast
                if idx==length:
                    terminal_var=log_sum_exp(forward_var+self.transitions[self.tag_to_ix[self.END_TAG]])
                    break
            score+=terminal_var
        return score/slot_features.size(0)

    def _score_given_sentence_fast(self,batch_feats,batch_tags,input_length):
        pass


    def _score_given_sentence(self,batch_feats,batch_tags,input_length):
        score=Variable(torch.zeros(1))
        append_tag=Variable(torch.LongTensor([self.tag_to_ix[self.START_TAG]]*batch_tags.size(0)).unsqueeze(dim=1))
        if self.cuda_:
            score=score.cuda()
            append_tag=append_tag.cuda()
        # append START_TAG
        batch_tags=torch.cat([append_tag,batch_tags],dim=1)
        for i,feats in enumerate(batch_feats):
            tags=batch_tags[i]
            length=input_length[i]
            for idx,feat in enumerate(feats):
                score=score+self.transitions[tags[idx+1],tags[idx]]+feat[tags[idx+1]]
                idx+=1
                if idx==length:
                    break
            score=score+self.transitions[self.tag_to_ix[self.END_TAG],tags[length].data[0]]
        return score/batch_feats.size(0)

    def forward(self,input_data,input_length,ignore_slot_index=-100):
        slot_features,sentence_results=self._rnn_features(input_data,input_length)        
        _,tag_seq=self._viterbi_decode(slot_features,input_length,ignore_slot_index)
        # _,tag_seq=self._viterbi_decode_fast(slot_features,input_length,ignore_slot_index)
        # tag_seq:[batch_size,len(seq)]
        _,sentence_results=torch.max(sentence_results,dim=1)
        # sentence_results:[batch_size]
        return tag_seq,sentence_results

    def _viterbi_decode_fast(self,batch_feats,input_length,ignore_slot_index=-100):
        pass

    def _viterbi_decode(self,batch_feats,input_length,ignore_slot_index=-100):
        score_result,id_result=[],[]
        max_length=input_length[0]
        init_vvars=torch.ones(self.slot_tagset_size)
        init_vvars=init_vvars*-1000000.
        init_vvars[self.tag_to_ix[self.START_TAG]]=0.
        padding_idx=Variable(torch.LongTensor([ignore_slot_index]))
        if self.cuda_:
            init_vvars=init_vvars.cuda()
            padding_idx=padding_idx.cuda()
        # init_vvars=init_vvars.unsqueeze(dim=1) #[len(slot_tag_size),1]
        for i,feats in enumerate(batch_feats):
            # batch_feats:[batch_size,len(slot_tag_size)]
            length=input_length[i]
            back_pointers=[] #用来回溯tag
            viterbi_var=Variable(init_vvars) # [tagset]
            for idx,feat in enumerate(feats):
                # feat:[len(slot_tag_size)]
                viterbi_var=viterbi_var+self.transitions #broadcast
                viterbi_var,max_idx=viterbi_var.max(dim=1)
                viterbi_var+=feat # 下一次迭代的viterbi variable
                # 处理标签回溯序列
                back_pointers.append(max_idx)
                idx+=1
                if idx==length:
                    break
            # transition to stop tag
            terminal_vvar=viterbi_var+self.transitions[self.tag_to_ix[self.END_TAG]]
            path_score,best_id=terminal_vvar.max(dim=0)
            score_result.append(path_score)
            # 回溯标签序列
            best_path=[best_id]
            for index in reversed(back_pointers):
                best_id=index[best_id]
                best_path.append(best_id)
            start=best_path.pop()
            assert start.data[0]==self.tag_to_ix[self.START_TAG]
            best_path.reverse()
            best_path=best_path+[padding_idx]*(max_length-length)
            best_path=torch.cat(best_path,dim=0)
            id_result.append(best_path)
        return torch.cat(score_result,dim=0),torch.stack(id_result,dim=0)

    def load_module(self,load_path):
        self.load_state_dict(torch.load(open(load_path,'rb')))

    def save_module(self,save_path_name):
        torch.save(self.state_dict(),open(save_path_name,'wb'))