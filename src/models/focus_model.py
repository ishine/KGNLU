#!/usr/bin/env python
#coding=utf8
import os,sys,json,re
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence 

class FocusModel(nn.Module):

    def __init__(self,args,voc_size,slot_size,sentence_size):
        super(FocusModel,self).__init__()
        self._parse_namespace(args)
        self.num_directions=2 if self.bidirection else 1
        self.voc_size=voc_size
        self.slot_size=slot_size #含有 <BEOS> 的起始 TAG
        self.sentence_size=sentence_size

        self.word_embeddings=nn.Embedding(
            num_embeddings=self.voc_size+1, #加一由于padding_idx
            embedding_dim=self.embedding_size,
            padding_idx=self.voc_size
        )
        self.slot_embeddings=nn.Embedding(
            num_embeddings=self.slot_size+1, #加一由于padding_idx
            embedding_dim=self.embedding_size,
            padding_idx=self.slot_size
        )

        self.input_dropout_layer=nn.Dropout(p=self.input_dropout)

        self.encoder=getattr(nn,self.cell.upper())(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirection
        ) 
        self.decoder=getattr(nn,self.cell.upper())(
            input_size=self.embedding_size+self.hidden_size*self.num_directions if not self.post_attention else self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False
        )

        # if self.bidirection: #双向encoder转成单向decoder的线性变换
        #     self.half_size=nn.Linear(self.hidden_size*2,self.hidden_size)

        self.hidden2slot=nn.Linear(self.hidden_size*1,self.slot_size) if not self.post_attention else \
        nn.Linear(self.hidden_size*(1+self.num_directions),self.slot_size)
        self.hidden2class=nn.Linear(self.hidden_size*self.num_directions,self.sentence_size) 

    def _parse_namespace(self,namespace):
        self.input_dropout=namespace.input_dropout #输入dropout率
        self.dropout=namespace.dropout #中间层dropout率
        self.bidirection=namespace.bidirection #encoder是否用双向
        self.layers=namespace.layers #encoder和decoder的层数
        self.cell=namespace.cell.lower() # 使用LSTM还是GRU单元
        self.use_pretrained_embeddings=namespace.use_word_embeddings #是否使用预训练好的embedding，暂不支持
        self.embedding_size=namespace.embedding_size #embedding size
        self.hidden_size=namespace.hidden_size #隐层数目
        self.post_attention=namespace.post_attention #是否在decoder RNN输出后再利用attention
        self.cuda_=namespace.cuda #是否用cuda

    def init_encoder_hidden_states(self,input_data):
        batch_size=input_data.size(0) if self.encoder.batch_first else input_data.size(1)
        h0=Variable(torch.zeros(self.layers*self.num_directions,batch_size,self.hidden_size),requires_grad=False)
        c0=Variable(torch.zeros(self.layers*self.num_directions,batch_size,self.hidden_size),requires_grad=False)
        if self.cuda_:
            h0=h0.cuda()
            c0=c0.cuda()
        return (h0,c0)

    def get_encoder_output(self,input_data,length_list):
        # encoder part
        # output:[batch_size,max_seq_length,hidden_size*num_directions]
        # ht,ct:[layers*num_directions,batch_size,hidden_size]
        if self.cell=='lstm':
            enc_h0_c0=self.init_encoder_hidden_states(input_data)
        else:
            enc_h0_c0,_=self.init_encoder_hidden_states(input_data)
        embeds=self.input_dropout_layer(self.word_embeddings(input_data))
        packed_embeds=pack_padded_sequence(embeds,length_list,batch_first=True)
        unpacked_out,enc_h0_c0=self.encoder(packed_embeds,enc_h0_c0)
        output,_=pad_packed_sequence(unpacked_out,batch_first=True)
        return output,enc_h0_c0

    def get_decoder_output(self,encoder_output,hidden_state,slot_tags,length_list):
        # encoder_output: [batch_size,max_seq_length,hidden_size*num_directions]
        # hidden_state: ht_ct:[layers*1,batch_size,hidden_size]
        # slot_tags:[batch_size,max_seq_length]
        slot_embeds=self.input_dropout_layer(self.slot_embeddings(slot_tags)) 
        # slot_embeds: [batch_size,max_seq_length,embedding_size]
        if not self.post_attention:
            # slot_tags和encoder_output拼接得到输入
            decoder_input=torch.cat([self.input_dropout_layer(encoder_output),slot_embeds],dim=2)
        else:
            decoder_input=slot_embeds
        packed_decoder_input=pack_padded_sequence(decoder_input,length_list,batch_first=True)
        unpacked_out,ht_ct=self.decoder(packed_decoder_input,hidden_state)
        out,_=pad_packed_sequence(unpacked_out,batch_first=True) 
        # out: [batch_size,max_seq_length,hidden_size]
        if not self.post_attention:
            slot_result=F.log_softmax(self.hidden2slot(out),dim=2) 
            # slot_result: [batch_size,max_seq_length,slot_size]
        else:
            out=torch.cat([self.input_dropout_layer(encoder_output),out],dim=2)
            slot_result=F.log_softmax(self.hidden2slot(out),dim=2)
        return slot_result,ht_ct

    def teacher_force_training(self,input_data,length_list,slot_tags):
        # encoder part
        if self.cell=='lstm':
            output,(enc_ht,enc_ct)=self.get_encoder_output(input_data,length_list)
        else:
            output,enc_ht=self.get_encoder_output(input_data,length_list)
        # output:[batch_size,max_seq_length,hidden_size*num_directions]
        # ht,ct:[layers*num_directions,batch_size,hidden_size]
        
        # sentence classification
        index_slices=[self.layers*2-2,self.layers*2-1] if self.bidirection else [self.layers*1-1]
        index_slices = Variable(torch.LongTensor(index_slices))
        if self.cuda_:
            index_slices= index_slices.cuda()
        sentence_features=torch.index_select(enc_ht,0,index_slices)
        sentence_features_ht=sentence_features.transpose(0,1)
        sentence_features_ht_reshape=sentence_features_ht.contiguous().view(sentence_features_ht.size(0),-1)
        sentence_results=F.log_softmax(self.hidden2class(sentence_features_ht_reshape),dim=1)

        #传递hidden state
        if self.bidirection:
            #如果encoder是双向，需要对hidden state的输出进行处理才能初始化decoder的hidden state
            # 方法1：只选取逆向的
            index_slices = [2*i+1 for i in range(self.layers)]  # generated from the reversed path
            index_slices = Variable(torch.LongTensor(index_slices))
            if self.cuda_:
                index_slices= index_slices.cuda()
            dec_ht = torch.index_select(enc_ht, 0, index_slices)
            if self.cell=='lstm':
                dec_ct = torch.index_select(enc_ct, 0, index_slices)
            # 方法2：两个方向的相加
            # index_slices=[2*i for i in range(self.layers)]
            # index_slices_reverse=[2*i+1 for i in range(self.layers)]
            # if self.cuda_:
            #     index_slices= index_slices.cuda()
            #     index_slices_reverse=index_slices.cuda()
            # dec_ht=torch.index_select(enc_ht,0,index_slices)+torch.index_select(enc_ht,0,index_slices_reverse)
            # if self.cell=='lstm':
            #     dec_ct=torch.index_select(enc_ct,0,index_slices)+torch.index_select(enc_ct,0,index_slices_reverse)
            # 方法3：两个方向拼接起来后经过一个linear层,self.half_size
            # dec_ht=self.half_size(enc_ht.contiguous().view(self.layers,output.size(0),self.hidden_size*self.num_directions))
            # if self.cell=='lstm':
            #     dec_ct=self.half_size(enc_ct.contiguous().view(self.layers,output.size(0),self.hidden_size*self.num_directions))
        else:
            dec_ht=enc_ht
            if self.cell=='lstm':
                dec_ct=enc_ct

        #decoder part
        if self.cell=='lstm':
            slot_results,_=self.get_decoder_output(output,(dec_ht,dec_ct),slot_tags,length_list)
        else:
            slot_results,_=self.get_decoder_output(output,dec_ht,slot_tags,length_list)

        return slot_results,sentence_results

    def decoder_greed(self,input_data,length_list,init_tags=0,ignore_slot_index=-100): 
        # 每次都从输出找出最大概率的标签作为下一个阶段的输入
        # init_tags存储<BEOS>的idx,用来初始化decoder的输入
        
        # encoder 部分
        if self.cell=='lstm':
            encoder_output,(enc_ht,enc_ct)=self.get_encoder_output(input_data,length_list)
        else:
            encoder_output,enc_ht=self.get_encoder_output(input_data,length_list)
        # encoder_output:[batch_size,max_seq_length,hidden_size*num_directions]
        # ht,ct:[layers*num_directions,batch_size,hidden_size]
        
        # sentence classification
        index_slices=[self.layers*2-2,self.layers*2-1] if self.bidirection else [self.layers*1-1]
        index_slices = Variable(torch.LongTensor(index_slices))
        if self.cuda_:
            index_slices= index_slices.cuda()
        sentence_features=torch.index_select(enc_ht,0,index_slices)
        sentence_features_ht=sentence_features.transpose(0,1)
        sentence_features_ht_reshape=sentence_features_ht.contiguous().view(sentence_features_ht.size(0),-1)
        sentence_results=F.log_softmax(self.hidden2class(sentence_features_ht_reshape),dim=1) 
        # sentence_results: [batch_size,len(sentence_size)]
        
        # 传递hidden state
        if self.bidirection:
            #如果encoder是双向，需要对hidden state的输出进行处理才能初始化decoder的hidden state
            # 方法1：只选取逆向的
            index_slices = [2*i+1 for i in range(self.layers)]  # generated from the reversed path
            index_slices = Variable(torch.LongTensor(index_slices))
            if self.cuda_:
                index_slices= index_slices.cuda()
            dec_ht = torch.index_select(enc_ht, 0, index_slices)
            if self.cell=='lstm':
                dec_ct = torch.index_select(enc_ct, 0, index_slices)
            # 方法2：两个方向的相加
            # index_slices=[2*i for i in range(self.layers)]
            # index_slices_reverse=[2*i+1 for i in range(self.layers)]
            # if self.cuda_:
            #     index_slices= index_slices.cuda()
            #     index_slices_reverse=index_slices.cuda()
            # dec_ht=torch.index_select(enc_ht,0,index_slices)+torch.index_select(enc_ht,0,index_slices_reverse)
            # if self.cell=='lstm':
            #     dec_ct=torch.index_select(enc_ct,0,index_slices)+torch.index_select(enc_ct,0,index_slices_reverse)
            # 方法3：两个方向拼接起来后经过一个linear层,self.half_size
            # dec_ht=self.half_size(enc_ht.contiguous().view(self.layers,output.size(0),self.hidden_size*self.num_directions))
            # if self.cell=='lstm':
            #     dec_ct=self.half_size(enc_ct.contiguous().view(self.layers,output.size(0),self.hidden_size*self.num_directions))
        else:
            dec_ht=enc_ht
            if self.cell=='lstm':
                dec_ct=enc_ct        

        minibatch_size=input_data.size(0) if self.encoder.batch_first else input_data.size(1)
        max_length=input_data.size(1) if self.encoder.batch_first else input_data.size(0)

        # decoder 部分
        last_tags=Variable(torch.LongTensor([init_tags]*minibatch_size).unsqueeze(dim=1)) 
        # last_tags: [batch_size,1] 第一个输入给decoder网络的是<BEOS>特殊起始符号
        if self.cuda_:
            last_tags=last_tags.cuda()
        top_path,top_path_tag_scores = [],[]
        return_path= [0]*minibatch_size
        top_dec_h_t= [0]*minibatch_size #用来记录batch中每一个样本的输出dec_ht
        if self.cell=='lstm':
            top_dec_c_t=[0]*minibatch_size
        for i in range(max_length):
            # slot_embeds: [batch,1,embed_size]
            slot_embeds=self.input_dropout_layer(self.slot_embeddings(last_tags)) 
            # encoder_output:[batch_size,max_seq_length,hidden_size*num_directions]
            if not self.post_attention:
                # decoder_input:[batch_size,1,hidden_size*num_directions+embed_size]
                decoder_input=torch.cat([self.input_dropout_layer(encoder_output[:,i:i+1]),slot_embeds],dim=2)
            else:
                # decoder_input:[batch_size,1,embed_size]
                decoder_input=slot_embeds
            if self.cell=='lstm':
                decoder_output,(dec_ht,dec_ct)=self.decoder(decoder_input,(dec_ht,dec_ct))
            else:
                decoder_output,dec_ht=self.decoder(decoder_input,dec_ht)
            # decoder_output:[batch_size,1,hidden_size*1]
            # dec_ht,dec_ct:[layers*1,batch_size,hidden_size]
            if self.post_attention:
                decoder_output=torch.cat([self.input_dropout_layer(encoder_output[:,i:i+1]),decoder_output],dim=2)
            decoder_output_reshape=decoder_output.contiguous().view(decoder_output.size(0),-1) 
            # decoder_output_reshape: [batch_size,hidden_size] or [batch_size,hidden_size*(1+num_directions)]
            slot_scores=F.log_softmax(self.hidden2slot(decoder_output_reshape),dim=1) 
            # slot_scores: [batch_size,slot_size]
            top_path_tag_scores.append(torch.unsqueeze(slot_scores,dim=1)) #[batch_size,1,slot_size]
            _,max_idx=slot_scores.max(dim=1)
            last_tags=max_idx.unsqueeze(dim=1) # [batch_size,1]
            # 处理top path
            if i==0:
                top_path=last_tags # [batch_size,1]
            else:
                top_path=torch.cat([top_path,last_tags],dim=1) # [batch_size,i+1]
            for j in range(minibatch_size):
                if length_list[j]==i+1: 
                    # 保留tag完成的样例
                    if i+1<max_length:
                        padding_idx=Variable(torch.ones(max_length-length_list[j]).type(torch.LongTensor)*ignore_slot_index)
                        if self.cuda_:
                            padding_idx=padding_idx.cuda()
                        return_path[j]=torch.cat([top_path[j],padding_idx],dim=0)
                    else:
                        return_path[j]=top_path[j] # [max_length]
                    #如果此时batch里某一个样例达到了其sequence长度，保留dec_ht
                    top_dec_h_t[j]=dec_ht[:,j:j+1,:]
                    if self.cell=='lstm':
                        top_dec_c_t[j]=dec_ct[:,j:j+1,:] #保留最终的dec_ht和dec_ct
        top_path=torch.stack(return_path,dim=0) # top_path: [batch_size,max_length]
        top_path_tag_scores=torch.cat(top_path_tag_scores,dim=1) # [batch_size,max_seq_length,slot_size]
        top_dec_h_t=torch.cat(top_dec_h_t, dim=1)
        if self.cell=='lstm':
            top_dec_c_t=torch.cat(top_dec_c_t, dim=1)
            return top_path,top_path_tag_scores,(top_dec_h_t,top_dec_c_t),sentence_results
        else:
            return top_path,top_path_tag_scores,top_dec_h_t,sentence_results

    def decoder_beamer(self,input_data,length_list,beam_size,init_tags=0,ignore_slot_index=-100):  
        # 始终保留n个概率最大的输出路径
        # encoder部分
        if self.cell=='lstm':
            encoder_output,(enc_ht,enc_ct)=self.get_encoder_output(input_data,length_list)
        else:
            encoder_output,enc_ht=self.get_encoder_output(input_data,length_list)
        # encoder_output:[batch_size,max_seq_length,hidden_size*num_directions]
        # ht,ct:[layers*num_directions,batch_size,hidden_size]
        
        # sentence classification
        index_slices=[self.layers*2-2,self.layers*2-1] if self.bidirection else [self.layers*1-1]
        index_slices = Variable(torch.LongTensor(index_slices))
        if self.cuda_:
            index_slices= index_slices.cuda()
        sentence_features=torch.index_select(enc_ht,0,index_slices)
        sentence_features_ht=sentence_features.transpose(0,1)
        sentence_features_ht_reshape=sentence_features_ht.contiguous().view(sentence_features_ht.size(0),-1)
        sentence_results=F.log_softmax(self.hidden2class(sentence_features_ht_reshape),dim=1) #[batch_size,len(sentence_size)]
        
        #传递hidden state
        if self.bidirection:
            #如果encoder是双向，需要对hidden state的输出进行处理才能初始化decoder的hidden state
            # 方法1：只选取逆向的
            index_slices = [2*i+1 for i in range(self.layers)]  # generated from the reversed path
            index_slices = Variable(torch.LongTensor(index_slices))
            if self.cuda_:
                index_slices= index_slices.cuda()
            dec_ht = torch.index_select(enc_ht, 0, index_slices)
            if self.cell=='lstm':
                dec_ct = torch.index_select(enc_ct, 0, index_slices)
            # 方法2：两个方向的相加
            # index_slices=[2*i for i in range(self.layers)]
            # index_slices_reverse=[2*i+1 for i in range(self.layers)]
            # if self.cuda_:
            #     index_slices= index_slices.cuda()
            #     index_slices_reverse=index_slices.cuda()
            # dec_ht=torch.index_select(enc_ht,0,index_slices)+torch.index_select(enc_ht,0,index_slices_reverse)
            # if self.cell=='lstm':
            #     dec_ct=torch.index_select(enc_ct,0,index_slices)+torch.index_select(enc_ct,0,index_slices_reverse)
            # 方法3：两个方向拼接起来后经过一个linear层,self.half_size
            # dec_ht=self.half_size(enc_ht.contiguous().view(self.layers,output.size(0),self.hidden_size*self.num_directions))
            # if self.cell=='lstm':
            #     dec_ct=self.half_size(enc_ct.contiguous().view(self.layers,output.size(0),self.hidden_size*self.num_directions))            
        else:
            dec_ht=enc_ht
            if self.cell=='lstm':
                dec_ct=enc_ct   

        # beam search decode
        minibatch_size=input_data.size(0) if self.encoder.batch_first else input_data.size(1)
        max_length=input_data.size(1) if self.encoder.batch_first else input_data.size(0)
        # 初始化
        last_tags=Variable(torch.LongTensor([[init_tags]*minibatch_size]*beam_size).unsqueeze(dim=2)) # [beam,batch,1]
        last_ht=torch.stack([dec_ht]*beam_size,dim=0) # last_ht:[beam,layer*1,batch,hidden_size]
        batch_k_slot,batch_k_slot_scores=[],[]
        current_scores=Variable(torch.Tensor([[0]*beam_size]*minibatch_size).unsqueeze(dim=2))
        if self.cuda_:
            last_tags=last_tags.cuda()
            current_scores=current_scores.cuda() # [batch,beam_size,1]
        if self.cell=='lstm':
            last_ct=torch.stack([dec_ct]*beam_size,dim=0) # last_ct:[beam_size,layers*1,batch_size,hidden_size]
            return_ct=[0]*minibatch_size
        return_tag_list,return_tag_scores=[0]*minibatch_size,[0]*minibatch_size
        return_ht=[0]*minibatch_size
        # 开始forward
        for i in range(max_length):
            scores=[] # 记录当前的beam_size个scores
            dec_ht_list=[] # 记录当前的beam_size个dec_ht结果
            if self.cell=='lstm':
                dec_ct_list=[]
            for j in range(beam_size):
                # last_tags: [beam_size,batch_size,1]
                slot_embeds=self.input_dropout_layer(self.slot_embeddings(last_tags[j])) 
                # slot_embeds: [batch_size,1,embedding_size]
                if not self.post_attention:
                    # encoder_output:[batch_size,max_seq_length,hidden_size*num_directions]
                    # dencoder_input:[batch_size,1,hidden_size*num_directions+embedding_size]
                    decoder_input=torch.cat([self.input_dropout_layer(encoder_output[:,i:i+1]),slot_embeds],dim=2) 
                else:
                    # dencoder_input:[batch_size,1,embedding_size]
                    decoder_input=slot_embeds
                if self.cell=='lstm':
                    # dec_ht,dec_ct:[layers*1,batch_size,hidden_size]
                    decoder_output,(dec_ht,dec_ct)=self.decoder(decoder_input,(last_ht[j],last_ct[j]))
                    dec_ct_list.append(dec_ct)
                else: 
                    decoder_output,dec_ht=self.decoder(decoder_input,last_ht[j])
                dec_ht_list.append(dec_ht) 
                # decoder_output:[batch_size,1,hidden_size]
                if self.post_attention:
                    decoder_output=torch.cat([self.input_dropout_layer(encoder_output[:,i:i+1]),decoder_output],dim=2)
                slot_scores=F.log_softmax(self.hidden2slot(decoder_output),dim=1) 
                # slot_scores: [batch_size,1,slot_size]
                scores.append(slot_scores)
            # scores: beam_size 个 [batch_size,slot_size]
            tmp_scores=torch.cat(scores,dim=1) # tmp_scores: [batch,beam_size,slot_size]
            # current_scores: [batch_size,beam_size,1]
            current_scores=current_scores+tmp_scores # 广播计算
            current_scores=current_scores.contiguous().view(minibatch_size,-1)
            kbest_scores,kbest_idx=current_scores.topk(beam_size,dim=1,largest=True,sorted=True) 
            kbest_path_idx=kbest_idx/self.slot_size # 当前kbest分别以原来kbest中的哪些作为输入,用来选择路径、dec_ht和dec_ct
            kbest_slot_idx=kbest_idx%self.slot_size # kbest分别代表第几个输出tag
            # kbest_*: [batch_size,beam_size]
            
            # last_tag更新，current_score更新
            last_tags=kbest_slot_idx.transpose(0,1).unsqueeze(dim=2) # last_tags:[beam_size,batch_size,1]
            current_scores=kbest_scores.unsqueeze(dim=2) #current_scores:[batch_size,beam_size,1]
            
            # last_ht更新: beam_size 个 [layers*1,batch_size,hidden_size]
            dec_ht_list=torch.stack(dec_ht_list,dim=0) # [beam_size,layers*1,batch_size,hidden_size]
            index=kbest_path_idx.transpose(0,1).unsqueeze(dim=1).unsqueeze(dim=3).expand(dec_ht_list.size())
            # index: [beam_size,layers*1,batch_size,hidden_size]
            last_ht=dec_ht_list.gather(dim=0,index=index) # [beam_size,layers*1,batch_size,hidden_size]
            if self.cell=='lstm':
                dec_ct_list=torch.stack(dec_ct_list,dim=0)
                # index=kbest_path_idx.transpose(0,1).unsqueeze(dim=1).unsqueeze(dim=3).expand(dec_ct_list.size())
                last_ct=dec_ct_list.gather(dim=0,index=index)
            
            # tag_path路径添加
            if i==0:
                batch_k_slot=kbest_slot_idx.unsqueeze(dim=2) # batch_k_slot: [batch_size,beam_size,1]
            else:
                index=kbest_path_idx.unsqueeze(dim=2).expand(batch_k_slot.size())
                batch_k_slot=batch_k_slot.gather(dim=1,index=index) # [batch_size,beam_size,(i)]
                batch_k_slot=torch.cat([batch_k_slot,kbest_slot_idx.unsqueeze(dim=2)],dim=2) # [batch_size,beam_size,(i+1)]
            
            # tag_scores分数添加
            if i==0:
                batch_k_slot_scores=tmp_scores.unsqueeze(dim=2) # [batch,beam_size,1,slot_size]
            else:
                index=kbest_path_idx.unsqueeze(dim=2).unsqueeze(dim=3).expand(batch_k_slot_scores.size())
                batch_k_slot_scores=batch_k_slot_scores.gather(dim=1,index=index) # [batch,beam_size,(i),slot_size]
                index=kbest_path_idx.unsqueeze(dim=2).expand(tmp_scores.size()) # tmp_scores: [batch,beam_size,slot_size]
                tmp_scores=tmp_scores.gather(dim=1,index=index)
                batch_k_slot_scores=torch.cat([batch_k_slot_scores,tmp_scores.unsqueeze(2)],dim=2) #[batch,beam_size,(i+1),slot_size]
            
            # save finished result
            for s in range(minibatch_size):
                if length_list[s]==i+1:
                    # 保存已经完成的batch
                    if i+1<max_length:
                        padding_slot=Variable(torch.ones(1,beam_size,max_length-length_list[s]).type(torch.LongTensor)*ignore_slot_index)
                        padding_score=Variable(torch.log(torch.ones(1,beam_size,max_length-length_list[s],self.slot_size)/self.slot_size))
                        if self.cuda_:
                            padding_slot=padding_slot.cuda()
                            padding_score=padding_score.cuda()
                        return_tag_list[s]=torch.cat([batch_k_slot[s:s+1],padding_slot],dim=2) # [1,beam_size,max_length]
                        return_tag_scores[s]=torch.cat([batch_k_slot_scores[s:s+1],padding_score],dim=2) # [1,beam_size,max_length,slot_size]
                    else:
                        return_tag_list[s]=batch_k_slot[s:s+1]
                        return_tag_scores[s]=batch_k_slot_scores[s:s+1]
                    return_ht[s]=last_ht[:,:,s:s+1,:]
                    if self.cell=='lstm':
                        return_ct[s]=last_ct[:,:,s:s+1,:]

        # reorganize result
        return_tag_list=torch.cat(return_tag_list,dim=0) # [batch_size,beam_size,max_length]
        return_tag_scores=torch.cat(return_tag_scores,dim=0) # [batch_size,beam_size,max_length,slot_size]
        return_ht=torch.cat(return_ht,dim=2) # [beam_size,layers,batch_size,hidden_size]
        if self.cell=='lstm':
            return_ct=torch.cat(return_ct,dim=2)

        if self.cell=='lstm':
            return return_tag_list,return_tag_scores,(return_ht,return_ct),sentence_results
        else:
            return return_tag_list,return_tag_scores,return_ht,sentence_results

    def load_module(self,load_path):
        self.load_state_dict(torch.load(open(load_path,'rb')))

    def save_module(self,save_path_name):
        torch.save(self.state_dict(),open(save_path_name,'wb'))    