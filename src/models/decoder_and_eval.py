#!/usr/bin/env python3
#coding=utf8
import os,sys,json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from utils import *
from global_vars import *

def naive_decoder(slot_output,sentence_output,length_list,ignore_slot_index=-100):
    '''   
    [description]: pick maxmimum value from each char
    Arguments:
        slot_output : [batch_size,max_length_sequence,probability distribution of size len(slot_dict)]
        sentence_output : [batch_size,probability distribution of size len(sentence_dict)]
        length_list : length for each variable sentence
    Keyword Arguments:
        ignore_slot_index : index symbol of padding (default: {-100})
    Return:
        max_slot_idx: [batch_size,max_sequence_length]
        max_label_idx: [batch_size]
    '''
    # Variable转Tensor
    slot_output=slot_output.data.cpu()
    sentence_output=sentence_output.data.cpu()
    # 一个max函数足够了
    _,max_slot_idx=torch.max(slot_output,dim=2)
    mask=torch.zeros_like(max_slot_idx).type(torch.LongTensor)
    length_matrix=[]
    bias_matrix=[]
    max_length=length_list[0]
    assert max_length==max_slot_idx.size(1)
    for each in length_list: #根据length_list将padding的部分换为ignore_slot_index
        tmp=[1]*each+[0]*(max_length-each)
        tmp_bias=[0]*each+[ignore_slot_index]*(max_length-each)
        length_matrix.append(tmp)
        bias_matrix.append(tmp_bias)
    length_matrix=torch.LongTensor(length_matrix)
    bias_matrix=torch.LongTensor(bias_matrix)
    max_slot_idx=torch.mul(max_slot_idx,length_matrix)+bias_matrix
    _,max_label_idx=torch.max(sentence_output,dim=1)
    return max_slot_idx,max_label_idx

def kbest_decoder(slot_output,sentence_output,length_list,ignore_slot_index=-100,kbest=1):
    '''
    [description] naive decoder, pick k best probability path, beam search
    Arguments:
        slot_output : [batch_size,max_length_sequence,probability distribution of size len(slot_dict)]
        sentence_output : [batch_size,probability distribution of size len(sentence_dict)]
        length_list : length for each variable sentence
    Keyword Arguments:
        ignore_slot_index {number}: index symbol of padding (default: {-100})
        kbest {number} : how many paths we pick (default: {1})
    Return:
        slot_result : [batch_size,kbest,max_sequence_length]
        sentence_result : [batch_size,kbest]
    '''
    slot_output=slot_output.data.cpu()
    sentence_output=sentence_output.data.cpu()
    max_length=length_list[0]
    slot_size=slot_output.size(2)
    assert max_length==slot_output.size(1)
    # 由于是logsoftmax，动规时用加法
    slot_result=[]
    for i,each_sample in enumerate(slot_output):
        current_length=length_list[i]
        kbest_list=[[] for j in range(kbest)]
        for idx,each_distribution in enumerate(each_sample):
            idx+=1
            if idx==1:
                current_score=each_distribution
            else:
                current_score=current_score+each_distribution # use broadcast strategy
            current_score=current_score.contiguous().view(-1)
            kbest_value,kbest_idx=current_score.topk(kbest,dim=0)
            current_score=kbest_value.unsqueeze(dim=1)
            k_path=kbest_idx/slot_size # LongTensor 不需要用//
            k_idx=kbest_idx%slot_size
            old_kbest_list=kbest_list
            kbest_list=[]
            for each_pair in zip(k_path,k_idx):
                kbest_list.append(old_kbest_list[each_pair[0]]+[each_pair[1]])
            
            if idx==current_length:
                for j in range(kbest):
                    kbest_list[j]=kbest_list[j]+[ignore_slot_index]*(max_length-current_length)
                break
        slot_result.append(kbest_list)
    slot_result=torch.LongTensor(slot_result)   # batch_size*kbest*max_sequence_length 
    # sentence_kbest_labels
    _,sentence_result=sentence_output.topk(kbest,dim=1) # batch_size*kbest
    return slot_result,sentence_result

def bio_decoder(slot_output,sentence_output,length_list,reverse_slot_dict,ignore_slot_index=-100,kbest=1,penalty=-1000000):
    '''
    [description] : consider the sequential of B-I    
    Arguments:
        slot_output : [batch_size,max_length_sequence,probability distribution of size len(slot_dict)]
        sentence_output : [batch_size,probability distribution of size len(sentence_dict)]
        length_list : length for each variable sentence
        reverse_slot_dict : dict mapping index to slot names
    Keyword Arguments:
        ignore_slot_index {number} : index symbol of padding (default: {-100})
        kbest {number} : how many paths we pick (default: {1})
        penalty {number} : penalty for I-label without starting with B-label
    Return:
        slot_result : [batch_size,kbest,max_sequence_length]
        sentence_result : [batch_size,kbest]
    '''
    slot_output=slot_output.data.cpu()
    sentence_output=sentence_output.data.cpu()
    max_length=length_list[0]
    slot_size=slot_output.size(2)
    assert max_length==slot_output.size(1) and slot_size==len(reverse_slot_dict)

    slot_result=[]
    penalty_bio=np.vectorize(bio_penalty)
    current_idx=torch.LongTensor(list(range(slot_size))) # [0,1,2,...] 
    # 方法大致与kbest_decoder相似，beam search，但在计算路径的和时将bio标签考虑进去
    # 如果当前标签为I-xxx，则前一个标签必须为B-xxx或I-xxx,否则广播求和时给一定的penalty
    for i,each_sample in enumerate(slot_output):
        current_length=length_list[i]
        kbest_list=[[] for j in range(kbest)]
        for idx,each_distribution in enumerate(each_sample):
            idx+=1
            if idx==1:
                current_score=each_distribution+torch.from_numpy(penalty_bio(-1,current_idx,reverse_slot_dict,penalty)).type(torch.FloatTensor)
            else:
                current_score=current_score+each_distribution+torch.from_numpy(penalty_bio(kprev_idx,current_idx,reverse_slot_dict,penalty)).type(torch.FloatTensor) # use broadcast strategy
            current_score=current_score.contiguous().view(-1)
            kbest_value,kbest_idx=current_score.topk(kbest,dim=0)
            current_score=kbest_value.unsqueeze(dim=1)
            k_path=kbest_idx/slot_size # LongTensor 不需要用//
            k_idx=kbest_idx%slot_size
            old_kbest_list=kbest_list
            kbest_list=[]
            for each_pair in zip(k_path,k_idx):
                kbest_list.append(old_kbest_list[each_pair[0]]+[each_pair[1]])
            
            kprev_idx=k_idx.unsqueeze(dim=1) # kbest*1

            if idx==current_length:
                for j in range(kbest):
                    kbest_list[j]=kbest_list[j]+[ignore_slot_index]*(max_length-current_length)
                break
        slot_result.append(kbest_list)
    slot_result=torch.LongTensor(slot_result)   # [batch_size*kbest*max_sequence_length]
    
    # sentence_kbest_labels
    _,sentence_result=sentence_output.topk(kbest,dim=1) # [batch_size*kbest]
    return slot_result,sentence_result

def ontology_decoder(slot_output,sentence_output,reverse_slot_dict,reverse_sentence_dict,length_list,ontology,ignore_slot_index=-100,kbest=1):
    '''    
    description: use info from ontology, to combine sentence classification and slots 
    Arguments:
        slot_output : [batch_size,max_length_sequence,probability distribution of size len(slot_dict)]
        sentence_output : [batch_size,probability distribution of size len(sentence_dict)]
        reverse_slot_dict : dict mapping index to slot names
        reverse_sentence_dict : dict mapping index to sentence labels
        length_list : length for each variable sentence
        ontology : class Ontology used in decoder
    Keyword Arguments:
        ignore_slot_index : index symbol of padding (default: {-100})
    Return:
        slot_result : [batch_size,kbest,kbest,max_sequence_length] # 第一个kbest对应sentence,第二个对应slot
        sentence_result : [batch_size,kbest]
    '''
    slot_output=slot_output.data.cpu()
    sentence_output=sentence_output.data.cpu()
    max_length=length_list[0]
    slot_size=slot_output.size(2)
    assert max_length==slot_output.size(1) and slot_size==len(reverse_slot_dict)

    slot_result=[]
    penalty_bio=np.vectorize(bio_penalty)
    current_idx=torch.LongTensor(list(range(slot_size))) # [0,1,2,...] 
    # 方法大致与kbest_decoder相似，beam search，但在计算路径的和时将bio标签考虑进去
    # 如果当前标签为I-xxx，则前一个标签必须为B-xxx或I-xxx,否则广播求和时给一定的penalty
    for i,each_sample in enumerate(slot_output):
        current_length=length_list[i]
        kbest_list=[[] for j in range(kbest)]
        for idx,each_distribution in enumerate(each_sample):
            idx+=1
            if idx==1:
                current_score=each_distribution+torch.from_numpy(penalty_bio(-1,current_idx,reverse_slot_dict,penalty)).type(torch.FloatTensor)
            else:
                current_score=current_score+each_distribution+torch.from_numpy(penalty_bio(kprev_idx,current_idx,reverse_slot_dict,penalty)).type(torch.FloatTensor) # use broadcast strategy
            current_score=current_score.contiguous().view(-1)
            kbest_value,kbest_idx=current_score.topk(kbest,dim=0)
            current_score=kbest_value.unsqueeze(dim=1)
            k_path=kbest_idx/slot_size # LongTensor 不需要用//
            k_idx=kbest_idx%slot_size
            old_kbest_list=kbest_list
            kbest_list=[]
            for each_pair in zip(k_path,k_idx):
                kbest_list.append(old_kbest_list[each_pair[0]]+[each_pair[1]])
            
            kprev_idx=k_idx.unsqueeze(dim=1) # kbest*1

            if idx==current_length:
                for j in range(kbest):
                    kbest_list[j]=kbest_list[j]+[ignore_slot_index]*(max_length-current_length)
                break
        slot_result.append(kbest_list)
    slot_result=torch.LongTensor(slot_result)   # [batch_size*kbest*max_sequence_length]
    
    # sentence_kbest_labels
    _,sentence_result=sentence_output.topk(kbest,dim=1) # [batch_size*kbest]
    return slot_result,sentence_result   

def evaluation_char_level(slot_output,sentence_output,slot_ref,sentence_ref,reverse_slot_dict,reverse_sentence_dict,
    kbest=True,use_ontology=False,ignore_slot_index=-100,slot_dict=None,sentence_dict=None):
    '''
    [description]
        Evaluation for each slot and label
    Arguments:
        slot_output {[type]} -- [description] [batch_size,(kbest,kbest,)max_sequence_length]
        sentence_output {[type]} -- [description] [batch_size,(kbest)]
        slot_ref {[type]} -- [description] [batch_size,max_sequence_length]
        sentence_ref {[type]} -- [description] [batch_size]
        reverse_slot_dict {[dict]} -- [description] maps idx to slot labels
        reverse_sentence_dict {[dict]} -- [description] maps idx to sentence labels
    
    Keyword Arguments:
        ignore_slot_index {number} -- [description] (default: {-100})
        kbest {bool} -- [description] (default: {True})
        use_ontology {bool} -- [description] (default: {True})
        slot_dict {dict} 
        sentence_dict {dict}
    '''
    if ignore_slot_index not in reverse_slot_dict:
        reverse_slot_dict[ignore_slot_index]='ignore_slot_index'
    if not slot_dict:
        slot_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0}) #分别表示真实标签为x的个数，预测标签为x的个数，真实为x且预测为x的个数，真实不为x但预测为x的个数
    if not sentence_dict:
        sentence_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0})
    #只选取一个结果
    if use_ontology:
        kbest=True
    if use_ontology:
        slot_output=slot_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
    if kbest:
        sentence_output=sentence_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
        slot_output=slot_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
    assert slot_output.size()==slot_ref.size() and sentence_output.size()==sentence_ref.size()

    label_compare=np.vectorize(compare_label)
    label_compare(slot_output,slot_ref,slot_dict,reverse_slot_dict)
    label_compare(sentence_output,sentence_ref,sentence_dict,reverse_sentence_dict)
    if 'ignore_slot_index' in slot_dict:
        del slot_dict['ignore_slot_index']
    if ignore_slot_index in reverse_slot_dict:
        del reverse_slot_dict[ignore_slot_index]
    return slot_dict,sentence_dict

def evaluation_slot_level(slot_output,sentence_output,raw_data,semantics,reverse_slot_dict,reverse_sentence_dict,ontology,
    kbest=True,use_ontology=False,ignore_slot_index=-100,slot_dict=None,sentence_dict=None,fp=None,error_fp=None,post_processor=None):
    if not slot_dict:
        #分别表示真实标签为x的个数，预测标签为x的个数，真实为x且预测为x的个数，真实不为x但预测为x的个数
        # precision=TP/P,recall=TP/T,fscore=2*precision*recall/(precision+recall)
        slot_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0}) 
    if not sentence_dict:
        # 'TP'表示完全解析正确,'SubSet'表示解析的slot是原slot的子集，'SuperSet'表示解析的slot是原slot的超集
        sentence_dict={'Num':0,'TP':0,'SubSet':0,'SuperSet':0} 
    # 目前的做法是kbest只选取第一个结果
    if use_ontology:
        kbest=True
    if use_ontology:
        slot_output=slot_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
    if kbest:
        sentence_output=sentence_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
        slot_output=slot_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
    assert len(raw_data)==slot_output.size(0) and len(semantics)==len(raw_data) and sentence_output.dim()==1 and slot_output.dim()==2
    # slot level evaluation
    for i,each in enumerate(slot_output):
        current_semantic={'return_type':'','mainEntity':'','slots':[]}
        current_semantic['return_type']=reverse_sentence_dict[int(sentence_output[i])]
        bio_list=[]
        for each_char in each:
            if each_char==ignore_slot_index:
                break
            else:
                bio_list.append(reverse_slot_dict[int(each_char)])
        assert len(bio_list)==len(raw_data[i])
        node_info=[]
        current_label={'op':'eq','name':'','value':[]}
        for each_pair in zip(raw_data[i],bio_list):
            if not (each_pair[1].startswith('B-') or each_pair[1].startswith('I-')):#=='O':
                if current_label['name']!='': #上一个标签结束了
                    current_label['value']=reverse_preproc(current_label['value'])
                    node_info.append(current_label)
                    current_label={'op':'eq','name':'','value':[]}
            elif each_pair[1].startswith('B-'):
                if current_label['name']!='':
                    current_label['value']=reverse_preproc(current_label['value'])
                    node_info.append(current_label)
                    current_label={'op':'eq','name':'','value':[]}
                current_label['name']=each_pair[1].strip('B-')
                current_label['value'].append(each_pair[0])
            else: # label startswith I-
                if current_label['name']!='':
                    if current_label['name']==each_pair[1].strip('I-'):
                        current_label['value'].append(each_pair[0])
                    else:
                        print('[Warning]:mismatch B-label and I-label in',reverse_preproc(raw_data[i]),':',current_label['name'],'=>',each_pair[1].strip('I-'),'(',each_pair[0],')')
                        current_label['value']=reverse_preproc(current_label['value'])
                        node_info.append(current_label)
                        current_label={'op':'eq','name':'','value':[]}
                        current_label['name']=each_pair[1].strip('I-')
                        current_label['value'].append(each_pair[0])
                else:
                    print('[Warning]:bio label starts with I- in',reverse_preproc(raw_data[i]),':',each_pair[0],'=>',each_pair[1])
                    current_label['name']=each_pair[1].strip('I-')
                    current_label['value'].append(each_pair[0])
        if current_label['name']!='':
            current_label['value']=reverse_preproc(current_label['value'])
            node_info.append(current_label)
        current_semantic['slots']=node_info
        # 比较解析的semantic和semantic_ref
        semantic_ref=semantics[i]['slots']
        for each_slot in node_info:
            slot_name=each_slot['name']
            slot_dict[slot_name]['P']+=1
            if each_slot in semantic_ref:
                slot_dict[slot_name]['TP']+=1
            else:
                slot_dict[slot_name]['FP']+=1
        for each_slot in semantic_ref:
            slot_dict[each_slot['name']]['T']+=1
        # 做句子检查时需要检查current_semantic的合理性
        current_semantic=check_logical_form(current_semantic,ontology)
        #constraint mappings归一化，对参考的semantic_ref也进行归一是为了防止众包标注任务时类似2014年和2014这样的个人原因
        if post_processor:
            current_semantic=post_processor.post_process_constraint(current_semantic,semantic_type='non_recursive')
            semantic_ref=post_processor.post_process_constraint(semantics[i],semantic_type='non_recursive')
            current_semantic=post_processor.post_process_entity(current_semantic,semantic_type='non_recursive')
            semantic_ref=post_processor.post_process_entity(semantic_ref,semantic_type='non_recursive')
        else:
            semantic_ref=semantics[i]
        sentence_dict['Num']+=1
        if current_semantic['return_type']==semantic_ref['return_type']:
            count=len(current_semantic['slots'])
            count_ref=len(semantic_ref['slots'])
            in_number,out_number=0,0
            for each_slot in current_semantic['slots']:
                if each_slot in semantic_ref['slots']:
                    in_number+=1
            if count==in_number and count==count_ref and current_semantic['mainEntity']==semantic_ref['mainEntity']:
                sentence_dict['TP']+=1
            elif error_fp:
                error_fp.write('Query:'+reverse_preproc(raw_data[i])+'\n')
                error_fp.write('RefResult:'+json.dumps(semantic_ref,ensure_ascii=False)+'\n')
                error_fp.write('Parsed:'+json.dumps(current_semantic,ensure_ascii=False)+'\n\n')
            if in_number==count_ref and count>count_ref:
                sentence_dict['SuperSet']+=1
            elif count==in_number and count<count_ref:
                sentence_dict['SubSet']+=1
        elif error_fp:
            error_fp.write('Query:'+reverse_preproc(raw_data[i])+'\n')
            error_fp.write('RefResult:'+json.dumps(semantic_ref,ensure_ascii=False)+'\n')
            error_fp.write('Parsed:'+json.dumps(current_semantic,ensure_ascii=False)+'\n\n')
        if fp:
            fp.write('Query:'+reverse_preproc(raw_data[i])+'\n')
            fp.write('RefResult:'+json.dumps(semantic_ref,ensure_ascii=False)+'\n')
            fp.write('Parsed:'+json.dumps(current_semantic,ensure_ascii=False)+'\n\n')
    return slot_dict,sentence_dict

def obtain_semantics(slot_output,sentence_output,raw_data,reverse_slot_dict,reverse_sentence_dict,ontology,
    kbest=True,use_ontology=False,ignore_slot_index=-100,fp=None,post_processor=None,encapsulate=False):
    semantics_list=[]
    #只选取一个结果
    if use_ontology:
        kbest=True
    if use_ontology:
        slot_output=slot_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
    if kbest:
        sentence_output=sentence_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
        slot_output=slot_output.index_select(1,torch.LongTensor([0])).squeeze(dim=1)
    assert len(raw_data)==slot_output.size(0) and sentence_output.dim()==1 and slot_output.dim()==2
    # slot level evaluation
    for i,each in enumerate(slot_output):
        current_semantic={'return_type':'','mainEntity':'','slots':[]}
        current_semantic['return_type']=reverse_sentence_dict[int(sentence_output[i])]
        bio_list=[]
        for each_char in each:
            if each_char==ignore_slot_index:
                break
            else:
                bio_list.append(reverse_slot_dict[int(each_char)])
        assert len(bio_list)==len(raw_data[i])
        node_info=[]
        current_label={'op':'eq','name':'','value':[]}
        for each_pair in zip(raw_data[i],bio_list):
            if not (each_pair[1].startswith('B-') or each_pair[1].startswith('I-')):
                if current_label['name']!='':
                    current_label['value']=reverse_preproc(current_label['value'])
                    node_info.append(current_label)
                    current_label={'op':'eq','name':'','value':[]}
            elif each_pair[1].startswith('B-'):
                if current_label['name']!='':
                    current_label['value']=reverse_preproc(current_label['value'])
                    node_info.append(current_label)
                    current_label={'op':'eq','name':'','value':[]}
                current_label['name']=each_pair[1].strip('B-')
                current_label['value'].append(each_pair[0])
            else:
                if current_label['name']!='':
                    if current_label['name']==each_pair[1].strip('I-'):
                        current_label['value'].append(each_pair[0])
                    else:
                        print('[Warning]:mismatch B-label and I-label in',reverse_preproc(raw_data[i]),':',current_label['name'],'=>',each_pair[1].strip('I-'),'(',each_pair[0],')')
                        current_label['value']=reverse_preproc(current_label['value'])
                        node_info.append(current_label)
                        current_label={'op':'eq','name':'','value':[]}
                        current_label['name']=each_pair[1].strip('I-')
                        current_label['value'].append(each_pair[0])
                else:
                    print('[Warning]:bio label starts with I- in',reverse_preproc(raw_data[i]),':',each_pair[0],'=>',each_pair[1])
                    current_label['name']=each_pair[1].strip('I-')
                    current_label['value'].append(each_pair[0])
        if current_label['name']!='':
            current_label['value']=reverse_preproc(current_label['value'])
            node_info.append(current_label)
        current_semantic['slots']=node_info
        # 做句子检查时需要检查current_semantic的合理性
        current_semantic=check_logical_form(current_semantic,ontology)
        #constraint mappings归一化，对参考的semantic_ref也进行归一是为了防止众包标注任务时类似2014年和2014这样的个人原因
        if encapsulate:
            current_semantic=encapsulate_semantics(current_semantic,ontology)
            if post_processor:
                current_semantic=post_processor.post_process_constraint(current_semantic,semantic_type='recursive')
                current_semantic=post_processor.post_process_entity(current_semantic,semantic_type='recursive')
        elif post_processor:
            current_semantic=post_processor.post_process_constraint(current_semantic,semantic_type='non_recursive')
            current_semantic=post_processor.post_process_entity(current_semantic,semantic_type='non_recursive')
            
        semantics_list.append(current_semantic)
        if fp:
            fp.write("Query: "+' '.join(raw_data[i])+' =>\nParse: '+json.dumps(current_semantic,ensure_ascii=False)+'\n\n')
    return semantics_list

def compare_semantic(semantics_list,ref_semantics,slot_dict,sentence_dict,whole_dict,error_fp=None):
    for i,each_semantic in enumerate(semantics_list):
        ref_semantic=ref_semantics[i]
        whole_dict['Num']+=1
        if semantic_equal(each_semantic,ref_semantic):
            whole_dict['TP']+=1
        elif error_fp:
            error_fp.write('Parsed: '+json.dumps(each_semantic,ensure_ascii=False)+'\n')
            error_fp.write('Ref: '+json.dumps(ref_semantic,ensure_ascii=False)+'\n\n')
        parse_label,ref_label='nil','nil'
        for each in each_semantic:
            if each.startswith('er_'):
                each=each_semantic[each]
                parse_label=each["pred"] if each["expr_type"]=="rel_attr" else '${#'+each['pred']+'}'
                break
        for each in ref_semantic:
            if each.startswith('er_'):
                each=ref_semantic[each]
                ref_label=each["pred"] if each["expr_type"]=="rel_attr" else '${#'+each['pred']+'}'
                break
        sentence_dict[parse_label]['P']+=1
        sentence_dict[ref_label]['T']+=1
        if parse_label==ref_label:
            sentence_dict[parse_label]['TP']+=1
        else:
            sentence_dict[parse_label]['FP']+=1

        ref_slots=[]
        parse_slots=[]
        for each in ref_semantic:
            if each.startswith('ne_'):
                each=ref_semantic[each]
                ref_slots.append({'type':each['type'],'value':each['value']})
            elif each.startswith('er_'):
                each=ref_semantic[each]['filter']
                for each_slot in each:
                    if not each_slot['value'].startswith('ne_'):
                        ref_slots.append({'type':each_slot['constraint'],'value':each_slot['value']})
        for each in each_semantic:
            if each.startswith('ne_'):
                each=each_semantic[each]
                parse_slots.append({'type':each['type'],'value':each['value']})
            elif each.startswith('er_'):
                each=each_semantic[each]['filter']
                for each_slot in each:
                    if not each_slot['value'].startswith('ne_'):
                        parse_slots.append({'type':each_slot['constraint'],'value':each_slot['value']})

        for each in parse_slots:
            slot_dict[each['type']]['P']+=1
            if each in ref_slots:
                slot_dict[each['type']]['TP']+=1
            else:
                slot_dict[each['type']]['FP']+=1
        for each in ref_slots:
            slot_dict[each['type']]['T']+=1
    return slot_dict,sentence_dict,whole_dict

def semantic_equal(parse,ref):
    parse_predict,ref_predict='nil','nil'
    for each in parse:
        if each.startswith('er_'):
            parse_predict=parse[each]
            break
    for each in ref:
        if each.startswith('er_'):
            ref_predict=ref[each]
            break
    if parse_predict=='nil' and ref_predict=='nil':
        return parse==ref
    elif parse_predict!='nil' and ref_predict!='nil':
        if (parse_predict['pred']!=ref_predict['pred']) or (parse_predict['expr_type']!=ref_predict['expr_type']):
            return False
        if parse_predict['ent_idx'] and ref_predict['ent_idx']:
            # check mainEntity
            if parse[parse_predict['ent_idx']]!=ref[ref_predict['ent_idx']]:
                return False
        parse_filter,ref_filter=parse_predict['filter'],ref_predict['filter']
        parse_num,ref_num=len(parse_filter),len(ref_filter)
        if parse_num!=ref_num:
            return False
        in_num=0
        for each in ref_filter:
            if each['value'].startswith('ne_'):
                each['value']=ref[each['value']]['value']
        for each in parse_filter:
            if each['value'].startswith('ne_'):
                each['value']=parse[each['value']]['value']
            if each in ref_filter:
                in_num+=1
        if in_num==parse_num:
            return True
    else:
        return False


def bio_penalty(x,y,reverse_slot_dict,penalty=-1000000):
    x,y=int(x),int(y)
    if x<0:
        return penalty if reverse_slot_dict[y].startswith('I-') else 0
    if reverse_slot_dict[y].startswith('I-'):
        if reverse_slot_dict[x].startswith('O') or reverse_slot_dict[x]=='<BEOS>':
            return penalty
        elif reverse_slot_dict[x].startswith('B-') and reverse_slot_dict[x].lstrip('B-')!=reverse_slot_dict[y].lstrip('I-'):
            return penalty
        elif reverse_slot_dict[x].startswith('I-') and reverse_slot_dict[x]!=reverse_slot_dict[y]:
            return penalty
        else:
            return 0
    else:
        return 0

def ontology_penalty(sentence_labels,slot_labels,reverse_slot_dict,reverse_sentence_dict,ontology,punish=-0.1):
    # 根据当前句子分类的类别对结果进行删选
    # 如果分类是entity，则标签为O,B/I-attr/rel,B/I-rel_type,否则给一定的punish
    # 如果分类是attr，则标签为O,B/I-entity,否则给一定的punish
    # 如果分类是rel,则标签为O,B/I-entity(rel的entity),B/I-rel/attr(rel的entity的rel和attr)
    # 如果分类是nil，则只能有B/I-entity
    slot_label=reverse_slot_dict[slot_labels]
    if slot_label=='O':
        return 0
    else:
        slot_label=slot_label.lstrip('B-').lstrip('I-').strip('${#}')
    sentence_label=reverse_sentence_dict[sentence_labels]
    if sentence_label.startswith('${'):
        # ask concept
        sentence_label=sentence_label.strip('${#}')
        if slot_label in ontology.get_attr_and_rel(sentence_label):
            return 0
        elif slot_label in ontology.get_rel_type_list(sentence_label):
            return 0
        else:
            return punish
    elif sentence_label=='nil':
        if slot_label in list(ontology.entities):
            return 0
        else:
            return punish
    else:
        entity_list=ontology.get_entity_list(sentence_label)
        # ask relation or attribute
        if ontology.is_attribute(sentence_label):
            # attribute
            if slot_label in entity_list:
                return 0
            else:
                return punish
        else:
            # relation
            if slot_label in entity_list:
                return 0
            link_entity=ontology.get_type(sentence_label)
            if slot_label in ontology.get_attr_and_rel(link_entity):
                return 0
            elif slot_label in ontology.get_rel_type_list(link_entity):
                return 0
            else:
                return punish

def compare_label(x,ref_x,label_dict,reverse_label_dict):
    x=reverse_label_dict[x]
    ref_x=reverse_label_dict[ref_x]
    label_dict[x]['P']+=1
    label_dict[ref_x]['T']+=1
    if x==ref_x:
        label_dict[ref_x]['TP']+=1
    else:
        label_dict[x]['FP']+=1
    return None

def print_char_level_scores(slot_result_dict=None,sentence_result_dict=None,path=None):
    if slot_result_dict:
        for each in slot_result_dict:
            slot_result_dict[each]['precision']=slot_result_dict[each]['TP']/slot_result_dict[each]['P'] if slot_result_dict[each]['P']!=0 else 0.0
            slot_result_dict[each]['recall']=slot_result_dict[each]['TP']/slot_result_dict[each]['T'] if slot_result_dict[each]['T']!=0 else 0.0
            slot_result_dict[each]['fscore']=(2*slot_result_dict[each]['precision']*slot_result_dict[each]['recall'])/(slot_result_dict[each]['precision']+slot_result_dict[each]['recall']) \
            if (slot_result_dict[each]['precision']+slot_result_dict[each]['recall'])!=0 else 0.0
    if sentence_result_dict:
        for each in sentence_result_dict:
            sentence_result_dict[each]['precision']=sentence_result_dict[each]['TP']/sentence_result_dict[each]['P'] if sentence_result_dict[each]['P']!=0 else 0.0
            sentence_result_dict[each]['recall']=sentence_result_dict[each]['TP']/sentence_result_dict[each]['T'] if sentence_result_dict[each]['T']!=0 else 0.0
            sentence_result_dict[each]['fscore']=(2*sentence_result_dict[each]['precision']*sentence_result_dict[each]['recall'])/(sentence_result_dict[each]['precision']+sentence_result_dict[each]['recall']) \
            if (sentence_result_dict[each]['precision']+sentence_result_dict[each]['recall'])!=0 else 0.0
    if path:
        with open(path,'w') as of:
            if slot_result_dict:
                of.write('==================================\n\n')
                of.write('{0:25}{1:10}{2:10}{3:10}{4:10}{5:10}{6:10}{7:10}\n'.format('slot name','T','P','TP','FP','precision','recall','fscore'))
                macro_precision,macro_recall,true_macro_precision,true_macro_recall=0,0,0,0
                for each in sorted(slot_result_dict):
                    macro_precision+=slot_result_dict[each]['precision']
                    macro_recall+=slot_result_dict[each]['recall']
                    if each!='O' and each!='<BEOS>': # 不考虑O和<BEOS>标签
                        true_macro_precision+=slot_result_dict[each]['precision']
                        true_macro_recall+=slot_result_dict[each]['recall']
                    of.write('{0:25}{1:<10d}{2:<10d}{3:<10d}{4:<10d}{5:<10.2%}{6:<10.2%}{7:<10.2%}\n'.format(
                        each,slot_result_dict[each]['T'],slot_result_dict[each]['P'],slot_result_dict[each]['TP'],slot_result_dict[each]['FP'],
                        slot_result_dict[each]['precision'],slot_result_dict[each]['recall'],slot_result_dict[each]['fscore']))
                macro_precision=macro_precision/len(slot_result_dict)
                macro_recall=macro_recall/len(slot_result_dict)
                macro_fscore=2*macro_precision*macro_recall/(macro_precision+macro_recall) if (macro_precision+macro_recall)!=0 else 0.0
                true_macro_precision=true_macro_precision/(len(slot_result_dict)-1)
                true_macro_recall=true_macro_recall/(len(slot_result_dict)-1)
                true_macro_fscore=2*true_macro_precision*true_macro_recall/(true_macro_precision+true_macro_recall) if (true_macro_precision+true_macro_recall)>0 else 0
                of.write('-----------------------------------\n')
                of.write('{0:25}{1:<10.2%}{2:<10.2%}{3:<10.2%}\n'.format('macro_all',macro_precision,macro_recall,macro_fscore))
                of.write('{0:25}{1:<10.2%}{2:<10.2%}{3:<10.2%}\n'.format('true_macro_all',true_macro_precision,true_macro_recall,true_macro_fscore))
            
            if sentence_result_dict:
                of.write('\n==================================\n\n')
                
                of.write('{0:25}{1:10}{2:10}{3:10}{4:10}{5:10}{6:10}{7:10}\n'.format('sentence name','T','P','TP','FP','precision','recall','fscore'))
                macro_precision,macro_recall=0,0
                for each in sorted(sentence_result_dict):
                    macro_precision+=sentence_result_dict[each]['precision']
                    macro_recall+=sentence_result_dict[each]['recall']

                    of.write('{0:25}{1:<10d}{2:<10d}{3:<10d}{4:<10d}{5:<10.2%}{6:<10.2%}{7:<10.2%}\n'.format(each,
                        sentence_result_dict[each]['T'],sentence_result_dict[each]['P'],sentence_result_dict[each]['TP'],sentence_result_dict[each]['FP'],
                        sentence_result_dict[each]['precision'],sentence_result_dict[each]['recall'],sentence_result_dict[each]['fscore']))
                macro_precision=macro_precision/len(sentence_result_dict)
                macro_recall=macro_recall/len(sentence_result_dict)
                macro_fscore=2*macro_precision*macro_recall/(macro_precision+macro_recall)

                of.write('-----------------------------------\n')
                of.write('{0:25}{1:<10.2%}{2:<10.2%}{3:<10.2%}\n'.format('macro_all',macro_precision,macro_recall,macro_fscore))

def print_slot_level_scores(slot_result_dict=None,sentence_result_dict=None,path=None):
    if slot_result_dict:
        for each in slot_result_dict:
            slot_result_dict[each]['precision']=slot_result_dict[each]['TP']/slot_result_dict[each]['P'] if slot_result_dict[each]['P']!=0 else 0.0
            slot_result_dict[each]['recall']=slot_result_dict[each]['TP']/slot_result_dict[each]['T'] if slot_result_dict[each]['T']!=0 else 0.0
            slot_result_dict[each]['fscore']=(2*slot_result_dict[each]['precision']*slot_result_dict[each]['recall'])/(slot_result_dict[each]['precision']+slot_result_dict[each]['recall']) \
            if (slot_result_dict[each]['precision']+slot_result_dict[each]['recall'])!=0 else 0.0
    if sentence_result_dict:
        sentence_result_dict['accuracy']=sentence_result_dict['TP']/sentence_result_dict['Num']
    if path:
        with open(path,'w') as of:
            if slot_result_dict:
                of.write('==================================\n\n')  
                of.write('{0:25}{1:10}{2:10}{3:10}{4:10}{5:10}{6:10}{7:10}\n'.format('slot name','T','P','TP','FP','precision','recall','fscore'))
                macro_precision,macro_recall=0,0
                for each in sorted(slot_result_dict):
                    macro_precision+=slot_result_dict[each]['precision']
                    macro_recall+=slot_result_dict[each]['recall']
                    of.write('{0:25}{1:<10d}{2:<10d}{3:<10d}{4:<10d}{5:<10.2%}{6:<10.2%}{7:<10.2%}\n'.format(
                        each,slot_result_dict[each]['T'],slot_result_dict[each]['P'],slot_result_dict[each]['TP'],slot_result_dict[each]['FP'],
                        slot_result_dict[each]['precision'],slot_result_dict[each]['recall'],slot_result_dict[each]['fscore']))
                macro_precision=macro_precision/len(slot_result_dict)
                macro_recall=macro_recall/len(slot_result_dict)
                macro_fscore=2*macro_precision*macro_recall/(macro_precision+macro_recall) if (macro_precision+macro_recall)!=0 else 0.
                of.write('-----------------------------------\n')
                of.write('{0:25}{1:<10.2%}{2:<10.2%}{3:<10.2%}\n'.format('macro_all',macro_precision,macro_recall,macro_fscore))
            
            if sentence_result_dict:
                of.write('\n==================================\n\n')    
                of.write('sentence level scores:\n')
                of.write('total number={0:d}\n'.format(sentence_result_dict['Num']))
                of.write('TP={0:<10d}SubSet={1:<10d}SuperSet={2:<10d}\n'.format(sentence_result_dict['TP'],sentence_result_dict['SubSet'],sentence_result_dict['SuperSet']))
                of.write('accuracy={0:.2%}\n'.format(sentence_result_dict['accuracy']))

if __name__=='__main__':
    # current_idx=torch.LongTensor(list(range(45)))
    # penalty_bio=np.vectorize(bio_penalty)
    # slot_tag_dict,reverse_slot_dict=load_tagset(PATH_TO_SLOT_TAGSET['SpeechLab'])
    # print(torch.from_numpy(penalty_bio(-1,current_idx,reverse_slot_dict,-1000000)).type(torch.FloatTensor))
    parse='{"ne_0": {"value": "语音识别", "type": "concept"}, "er_1": {"expr_type": "entity", "pred": "company", "filter": [{"op": "eq", "constraint": "country", "value": "国内"}, {"op": "eq", "constraint": "core_technology", "value": "语音识别"}], "ent_idx": null}}'
    ref='{"ne_0": {"value": "语音识别", "type": "concept"}, "er_1": {"expr_type": "entity", "pred": "company", "filter": [{"op": "eq", "constraint": "core_technology", "value": "ne_0"}], "ent_idx": null}}'
    print(semantic_equal(parse,ref))