#!/usr/bin/env python
#coding=utf8

import os,sys,random,pickle,json,re
from collections import defaultdict,namedtuple
from global_vars import *
import torch
from torch.autograd import Variable
import random,copy
'''
[summary]
    Deal with data preprocessing and postprocessing functions
'''

def isenglish(c) :
    '''
    [description]
        Determine whehter a character is english word
    Returns:
        True or False
    '''
    c = c.lower()
    return c >= 'a' and c <= 'z'

def start_with_english_word(w):
    '''    
    [description]
        Determine whether a string starts with english word
    Arguments:
        w {str} -- [description]
    Returns:
        True or False
    '''
    w = w.lower()
    return w[0] >= 'a' and w[0] <= 'z'

def preproc(L) :
    '''
    [description]
        Format a string, split each non-english character and english word with a whitespace
    Arguments:
        L {str} -- [description]
    Returns:
        str -- format string
    '''
    L = L.lower().strip()
    L = re.sub(r'[ ][ ]+', ' ', L)
    idx = 0
    utter = []
    seg_flag = False
    while idx < len(L):
        if L[idx] == ' ':
            assert idx != 0 and idx != len(L) - 1
            if isenglish(L[idx - 1]) and isenglish(L[idx + 1]):
                seg_flag = True #utter.append('_')
            else:
                pass
        elif isenglish(L[idx]):
            if idx == 0:
                utter.append(L[idx])
            else:
                if isenglish(utter[-1][0]) and seg_flag == False:
                    utter[-1] += L[idx]
                else:
                    utter.append(L[idx])
                    seg_flag = False
        else:
            utter.append(L[idx])
        idx += 1
    
    return ' '.join(utter)

def reverse_preproc(words): 
    '''
    [description]
        Reverse a word list to a string,combine each non-english word,split english word with a whitespace
    Arguments:
        words {str list} -- [description]
    Returns:
        [str] -- [description]
    '''
    if len(words) == 0:
        return ""
    snt = words[0]
    for idx in range(0, len(words)-1):
        if start_with_english_word(words[idx]) and start_with_english_word(words[idx+1]):
            snt += ' ' + words[idx+1]
        else:
            snt += words[idx+1]
    return snt

def load_vocabulary(vocfile,ratio=1.0,beos=False):
    with open(vocfile,'r') as voc:
        voc_dict,reverse_vocdict,i={},{},0
        lines=voc.readlines()
        count=len(lines)*ratio
        for each in lines:
            each=each.strip()
            if each=='':
                continue
            else:
                i+=1 #当前load的词库大小，也可以理解为下一个tag的标签号
            if i>count:
                if beos:
                    voc_dict['<BOS>']=i
                    voc_dict['<EOS>']=i+1
                    reverse_vocdict[i]='<BOS>'
                    reverse_vocdict[i+1]='<EOS>'
                return voc_dict,reverse_vocdict
            features=each.split('\t')
            char,idx=features[0],features[1]
            voc_dict[char]=int(idx)
            reverse_vocdict[int(idx)]=char
        if beos:
            voc_dict['<BOS>']=i
            voc_dict['<EOS>']=i+1
            reverse_vocdict[i]='<BOS>'
            reverse_vocdict[i+1]='<EOS>'
    return voc_dict,reverse_vocdict

def load_tagset(path,seq2seq=False):
    with open(path,'r') as tag:
        tag_dict,reverse_tag_dict={},{}
        for each in tag:
            each=each.strip()
            if each=='':
                continue
            char,idx=each.split('\t')
            tag_dict[char]=int(idx)
            reverse_tag_dict[int(idx)]=char
        if seq2seq:
            tag_dict['<BEOS>']=len(tag_dict)-1
            reverse_tag_dict[len(tag_dict)-1]='<BEOS>'
    return tag_dict,reverse_tag_dict

def load_trainingdata(path_to_data,voc_dict,slot_tag_dict,sentence_tag_dict,beos=False,debug=False):
    '''    
    [return format]
    [['俞','凯'],[23,12],[11,0],17,json_semantics_dict]
    '''

    training_set=[]
    with open(path_to_data,'r') as infile:
        if beos:
            item=[['<BOS>'],[voc_dict['<BOS>']],[slot_tag_dict['O']],'','']
        else:
            item=[[],[],[],'','']
        for eachline in infile:
            eachline=eachline.strip()
            if eachline=='':
                if item[3]!='':
                    if beos:
                        item[0].append('<EOS>')
                        item[1].append(voc_dict['<EOS>'])
                        item[2].append(slot_tag_dict['O'])
                    training_set.append(item)
                if debug and len(training_set)==1000:
                    return training_set
                if beos:
                    item=[['<BOS>'],[voc_dict['<BOS>']],[slot_tag_dict['O']],'','']
                else:
                    item=[[],[],[],'','']
            elif item[3]=='':
                assert '\t' not in eachline
                item[3]=sentence_tag_dict[eachline]
            elif item[4]=='':
                assert 'return_type' in eachline and 'mainEntity' in eachline and 'slots' in eachline
                item[4]=json.loads(eachline)
            else:
                assert '\t' in eachline
                char,tag=eachline.split('\t')
                item[0].append(char)
                idx=voc_dict[char] if char in voc_dict else voc_dict['<UNK>'] # <UNK>
                item[1].append(idx)
                item[2].append(slot_tag_dict[tag])
        if item[3]!='':
            if beos:
                item[0].append('<EOS>')
                item[1].append(voc_dict['<EOS>'])
                item[2].append(slot_tag_dict['O'])
            training_set.append(item)
    return training_set

def load_evaldata(path_to_data,voc_dict,beos=False,stop_pattern=None,annotation=False):
    eval_sentences=[]
    eval_set=[]
    if stop_pattern:
        process=PreProcessor(stop_pattern)
    with open(path_to_data,'r') as infile:
        for line in infile:
            if line.strip()=='':
                continue
            if annotation:
                line,annotation=line.split('<=>')
                line,annotation=line.strip(),annotation.strip()
            if stop_pattern:
                line=process.preprocess(line.strip())
            if annotation:
                eval_sentences.append((preproc(line.lower()).split(' '),json.loads(annotation)))
            else:
                eval_sentences.append(preproc(line.lower()).split(' '))
    for each in eval_sentences:
        if annotation:
            annotation=each[1]
            each=each[0]
        if beos:
            item=[['<BOS>'],[voc_dict['<BOS>']]]
        else:
            item=[[],[]]
        for each_char in each:
            item[0].append(each_char)
            idx=voc_dict[each_char] if each_char in voc_dict else voc_dict['<UNK>']
            item[1].append(idx)
        if beos:
            item[0].append('<EOS>')
            item[1].append(voc_dict['<EOS>'])
        if annotation:
            item.append(annotation)
        eval_set.append(item)
    return eval_set

class PreProcessor():
    def __init__(self,stop_pat_file):
        self.stop_pat=self._load_stop_pattern(stop_pat_file)

    def _load_stop_pattern(self,stop_pat_file):
        pre_process=[]
        with open(stop_pat_file,'r') as infile:
            for line in infile:
                line=line.strip()
                if line=='' or line.startswith('#'):
                    continue
                else:
                    pre_process.append(line)
        return pre_process

    def rm_punc(self,query):
        pattern = "[\.\?\!\/_,$%^*(+\"\']+|[+——！，【】《》·。`？、“”~@#￥%……&*（）\[\]\{\}\:]+"
        return re.sub(pattern, '', query)

    def rm_prefix_and_suffix(self,query):
        for pat in self.stop_pat:
            query = re.sub(pat, '', query).strip()
        return query

    def preprocess(self,query):
        query=self.rm_punc(query.lower())
        return self.rm_prefix_and_suffix(query)

class PostProcessor():

    def __init__(self,constraint_mappings,entity_coreference_dir,ontology):

        self.constraint_mappings=self._build_constraint_mappings(constraint_mappings)
        self.entity_coreference=self._build_entity_coreherence(entity_coreference_dir)
        self.ontology=ontology # type=Ontology

    def _build_constraint_mappings(self,constraint_mappings):
        constraint_dict={}
        with open(constraint_mappings,'r') as infile:
            for line in infile:
                line=line.strip()
                if line=='':
                    continue
                constraint,value=line.split('=>')
                constraint,value=constraint.strip(),value.strip()
                if constraint not in constraint_dict:
                    constraint_dict[constraint]=[value]
                elif value in constraint_dict[constraint]:
                    continue
                else:
                    print('[Warning]:the same constraint:',constraint,'mapping to different values:',constraint_dict[constraint])
                    constraint_dict[constraint].append(value)
        return constraint_dict

    def _build_entity_coreherence(self,entity_coreference_dir):
        files=os.listdir(entity_coreference_dir)
        entity_dict={}
        for each in files:
            name=each[:each.find('.')] if '.' in each else each
            entity_dict['${#'+name+'}']={}
            with open(os.path.join(entity_coreference_dir,each),'r') as inf:
                for line in inf:
                    line=line.strip()
                    if line=='':
                        continue
                    entity,value=line.split('=>')
                    entity,value=entity.strip(),value.strip()
                    entity_dict['${#'+name+'}'][entity]=value
        return entity_dict

    def post_process_constraint(self,semantic,semantic_type='recursive'):
        if type(semantic)==str:
            semantic=json.loads(semantic)
            str_type=True
        else:
            str_type=False
        if semantic_type=='non_recursive': 
        # format: {'return_type':'','mainEntity':'','slots':[{'op':'','name':'','value':''},...]}
            for i,each in enumerate(semantic['slots']):
                if not each['name'].startswith('${#') and self.ontology.is_attribute(each['name']):# 判断是否是attr时没有指定entity，ontology的attr和rel不能重名
                    if each['value'] in self.constraint_mappings and len(self.constraint_mappings[each['value']])==1:
                        each['value']=self.constraint_mappings[each['value']][0] #没有冲突
                    elif each['value'] in self.constraint_mappings and len(self.constraint_mappings[each['value']])>1:
                        print('[Warning]:multiple constraint mapping values of',each['value'],'to',self.constraint_mappings[each['value']])
                        each['value']=self.constraint_mappings[each['value']][0] #有冲突的情况，暂时这么处理，以后可以在constr_mappings中加入对应的constraint名的信息
                    semantic['slots'][i]=each
        else:
        # format: {'ne_0':{'value':'','type':''},'er_1':{'expr_type':'','pred':'','filter':[{'op':'','constraint':'','value':''},...],'ent_idx':''}}
            for each in semantic:
                if each.startswith('ne'):
                    continue
                else:
                    assert each.startswith('er')
                    if self.ontology.is_relation(semantic[each]['pred']) or self.ontology.is_entity(semantic[each]['pred']):
                        filters=semantic[each]['filter']
                        for i,constraint in enumerate(filters):
                            if self.ontology.is_attribute(constraint['constraint']) and not constraint['value'].startswith('ne_'):
                                if constraint['value'] in self.constraint_mappings and len(self.constraint_mappings[constraint['value']])==1:
                                    constraint['value']=self.constraint_mappings[constraint['value']][0]
                                elif constraint['value'] in self.constraint_mappings and len(self.constraint_mappings[constraint['value']])>1:
                                    constraint['value']=self.constraint_mappings[constraint['value']][0]
                                    print('[Warning]:multiple constraint mapping values of',constraint['value'],'to',self.constraint_mappings[constraint['value']])
                                filters[i]=constraint
        return json.dumps(semantic,ensure_ascii=False) if str_type else semantic

    def post_process_entity(self,semantic,semantic_type='recursive'):
        if type(semantic)==str:
            semantic=json.loads(semantic)
            str_type=True
        else:
            str_type=False
        if semantic_type=='non_recursive':
        # format: {'return_type':'','mainEntity':'','slots':[{'op':'','name':'','value':''},...]}
            link_type=self.ontology.get_type(semantic['return_type']) if not semantic['return_type'].startswith('${#') and semantic['return_type']!='nil' else semantic['return_type'].strip('${#}')
            for i,each in enumerate(semantic['slots']):
                if each['name'].startswith('${#'):
                    if each['name'] in self.entity_coreference and each['value']==semantic['mainEntity']:
                        each['value']=self.entity_coreference[each['name']][each['value']] if each['value'] in self.entity_coreference[each['name']] else each['value']
                        semantic['mainEntity']=each['value']
                        semantic['slots'][i]=each
                    elif each['name'] in self.entity_coreference:
                        each['value']=self.entity_coreference[each['name']][each['value']] if each['value'] in self.entity_coreference[each['name']] else each['value']
                        semantic['slots'][i]=each                    
                elif self.ontology.is_relation(each['name']):# 更准确该用is_relation(each['name'],link_type)
                    link_entity='${#'+self.ontology.get_type(each['name'],link_type)+'}'
                    if link_entity in self.entity_coreference:
                        each['value']=self.entity_coreference[link_entity][each['value']] \
                    if each['value'] in self.entity_coreference[link_entity] else each['value']
                        semantic['slots'][i]=each
        else:
        # format: {'ne_0':{'value':'','type':''},'er_1':{'expr_type':'','pred':'','filter':[{'op':'','constraint':'','value':''},...],'ent_idx':''}}
            for item in semantic:
                if item.startswith('ne_'):
                    type_=semantic[item]['type']
                    value=semantic[item]['value']
                    if '${#'+type_+'}' in self.entity_coreference and value in self.entity_coreference['${#'+type_+'}']:
                        semantic[item]['value']=self.entity_coreference['${#'+type_+'}'][value]
        return json.dumps(semantic,ensure_ascii=False) if str_type else semantic

class Dataset():
    def __init__(self,data,cross,dev_size):
        self.data=data
        # DATA format:
        # each item: [['俞','凯'],[23,12],[11,0],17,semantics_dict] 
        # or [['俞','凯'],[23,12]]
        # or [['俞','凯'],[23,12],semantic_dict]
        self.length=len(data)
        # cross表示用几折交叉验证,不使用交叉验证cross=1
        self.cross=cross
        self.dev_size=dev_size
        if self.cross>1:
            self.data_split=self._split_data(self.data,self.cross)
        self.train_data=self.data[int(self.length*self.dev_size):]
        self.dev_data=self.data[:int(self.length*self.dev_size)]

    def _split_data(self,data,cross):
        # 将数据划分成cross份
        size=self.length//cross
        assert size!=0 and size>0
        data_split=[]
        start=0
        for i in range(int(cross)):
            if i+1==cross:
                chunk=data[start:]
            else:
                chunk=data[start:start+size]
            data_split.append(chunk)
            start+=size
        return data_split
        
    def select_dev_index(self,index):
        if self.cross<=1:
            return self.train_data,self.dev_data
        # 从k折交叉验证中选择第index个划分作为验证级，其余作为训练集(index从0,...,k-1)
        dev_data=self.data_split[index]
        train_data=[]
        for i,each_split in enumerate(self.data_split):
            if i!=index:
                train_data.extend(each_split)
        return train_data,dev_data

    def re_organize(self,minibatch,pad_num=0,ignore_slot_index=-100):
        # 将[(raw_data,data,slot_label,sentence_label,semantic_dict),...]
        # 转化为[raw_data_list,data_list,slot_label_list,sentence_label_list,semantic_dict_list],length_list并补齐
        raw_data_batch=[]
        data_batch=[]
        length_list=[]
        sorted_minibatch=sorted(minibatch,key=lambda x:len(x[0]),reverse=True)
        max_length=len(sorted_minibatch[0][0])
        if len(sorted_minibatch[0])>3: 
            #有标注
            slot_tag_batch=[]
            sentence_tag_batch=[]
            semantic_batch=[]
            for i,(raw_data,data,slot_tag,sentence_tag,semantic_dict) in enumerate(sorted_minibatch):
                size=len(raw_data)
                assert size==len(data) and size==len(slot_tag)
                length_list.append(int(size))
                raw_data_batch.append(raw_data)
                data_batch.append(data+[pad_num]*(max_length-size))
                slot_tag_batch.append(slot_tag+[ignore_slot_index]*(max_length-size))
                sentence_tag_batch.append(sentence_tag)
                semantic_batch.append(semantic_dict)
            data_batch=Variable(torch.LongTensor(data_batch))
            slot_tag_batch=Variable(torch.LongTensor(slot_tag_batch))
            sentence_tag_batch=Variable(torch.LongTensor(sentence_tag_batch))
            return ([raw_data_batch,data_batch,slot_tag_batch,sentence_tag_batch,semantic_batch],length_list)
        elif len(sorted_minibatch[0])==3:
            semantic_batch=[]
            for i,(raw_data,data,semantic_dict) in enumerate(sorted_minibatch):
                assert len(raw_data)==len(data)
                raw_data_batch.append(raw_data)
                data_batch.append(data+[pad_num]*(max_length-len(data)))
                semantic_batch.append(semantic_dict)
                length_list.append(int(len(raw_data)))
            data_batch=Variable(torch.LongTensor(data_batch))
            return ([raw_data_batch,data_batch,semantic_batch],length_list)
        else:
            # 无标注
            for i,(raw_data,data) in enumerate(sorted_minibatch):
                assert len(raw_data)==len(data)
                raw_data_batch.append(raw_data)
                data_batch.append(data+[pad_num]*(max_length-len(data)))
                length_list.append(int(len(raw_data)))
            data_batch=Variable(torch.LongTensor(data_batch))
            return ([raw_data_batch,data_batch],length_list)

    def get_mini_batches(self,batch_size=64,data=None,pad_num=0,ignore_slot_index=-100,shuffle=True):
        # each item in data should be: [['俞','凯'],[23,12],[11,0],17,json_semantics_dict] or [['俞','凯'],[23,12]]
        # pad_num是忽略的输入，用来补全变长输入
        # ignore_slot_index是忽略的slot输出，用来补全变长slot输出序列
        if not data:
            # 没有数据就用self.train_data
            data=self.train_data
        if shuffle:
            random.shuffle(data)
        mini_batch_list=[]
        start=0
        step=batch_size
        end=len(data)
        while start+step<end:
            tmp_minibatch=data[start:start+step]
            mini_batch_list.append(self.re_organize(tmp_minibatch,pad_num=pad_num,ignore_slot_index=ignore_slot_index))
            start+=step
        mini_batch_list.append(self.re_organize(data[start:],pad_num=pad_num,ignore_slot_index=ignore_slot_index))
        return mini_batch_list

class Ontology():
    '''    
    [description]
    Python encapsulation of ontology:
    Remember that relation or attribute names shouldn't conflict with entity names
    Remember that relation and attribute names shouldn't conflict with each other
    Different entities can have the same relation names, they can also have the same attribute names 
    '''
    def __init__(self,json_obj):
        if type(json_obj)==dict:
            self.ontology=json_obj
        else:
            self.ontology=json.load(open(json_obj,'r')) if type(json_obj)==str else json.load(json_obj)
        # self.entities is re-organization of self.ontology
        self.entities={}
        self._reOrganize()

    def _reOrganize(self):
        for each in self.ontology:
            self.entities[each]={'relations':{},'attributes':{}}
            for each_item in self.ontology[each]:
                if self.ontology[each][each_item] in ['string','float','int']:
                    self.entities[each]['attributes'][each_item]=self.ontology[each][each_item]
                else:
                    self.entities[each]['relations'][each_item]=self.ontology[each][each_item]

    def is_entity(self,entity_name):
        # determine whether a string is entity name
        if entity_name.startswith('${#') or entity_name.startswith('${.'):
            return entity_name[3:-1] in self.ontology
        return entity_name in self.ontology

    def is_relation(self,relation_name,entity=None):
        # determine whether relation_name is a relation (of entity)
        if entity:
            if entity not in self.entities:
                return False
            else:
                return relation_name in self.entities[entity]['relations']
        else:
            for each in self.entities:
                if relation_name in self.entities[each]['relations']:
                    return True
            return False

    def is_attribute(self,attribute_name,entity=None):
        # determine whether attribute_name is an attribute (of entity)
        if entity:
            if entity not in self.entities:
                return False
            else:
                return attribute_name in self.entities[entity]['attributes']
        else:
            for each in self.entities:
                if attribute_name in self.entities[each]['attributes']:
                    return True
            return False

    def get_type(self,relation_or_attr,entity=None):
        if not entity:
            for each in self.ontology:
                if relation_or_attr in self.ontology[each]:
                    return self.ontology[each][relation_or_attr]
            print('[Warning]:not found relation or attribute name in ontology!')
            return None
        else:
            if entity.strip('${#}') not in self.ontology:
                print('[Warning]:not found entity',entity,'in ontology!')
                return None
            if relation_or_attr in self.ontology[entity.strip('${#}')]:
                return self.ontology[entity.strip('${#}')][relation_or_attr]
            else:
                print('[Warning]:not found relation or attribute',relation_or_attr,'in entity',entity)
                return None

    def get_entity_list(self,relation_or_attr):
        # return the entity list where relation_or_attr lies
        entity_list=set()
        for each in self.ontology:
            if relation_or_attr in self.ontology[each]:
                entity_list.add(each)
        return list(entity_list)

    def get_attrs_and_rels(self,entity,choices=None):
        if entity not in self.ontology:
            return None
        if not None:
            return list(self.ontology[entity])
        elif choices.lower().startswith('attr'):
            return list(self.entities[entity]['attributes'])
        elif choices.lower().startswith('rel'):
            return list(self.entities[entity]['relations'])

    def get_rel_name_list(self,entity,rel_type=None):
        # 返回一个entity的所有关系的名字列表
        if entity not in self.ontology:
            return None
        if not rel_type:
            return self.get_attrs_and_rels(entity,choices='rel')
        else:
            return [each for each in self.get_attrs_and_rels(entity,choices='rel') if self.ontology[entity][each]==rel_type]

    def get_rel_type_list(self,entity,rel=None):
        # 返回一个entity的所有关系的类型列表
        if entity not in self.ontology:
            return []
        if rel and rel in self.ontology[entity]:
            return [self.get_type(rel,entity)]
        elif rel and rel not in self.ontology[entity]:
            return []
        s=set()
        for each in self.ontology[entity]:
            if self.is_relation(each,entity):
                s.add(each)
        return list(s)

def check_logical_form(semantic,ontology):
    if type(semantic)==str:
        semantic=json.loads(semantic)
        str_flag=True
    else:
        str_flag=False
    #{"return_type": "description", "mainEntity": "", "slots": [{"op": "eq", "name": "${#institute}", "value": "实验室"}]}
    sentence_tag=semantic['return_type']
    slots=semantic['slots']
    if sentence_tag.startswith('${#'):
        mainEntity=sentence_tag.strip('${#}')
        for i,item in enumerate(slots):
            if ontology.is_attribute(item['name']):
                if not ontology.is_attribute(item['name'],mainEntity):
                    slots[i]={}
            elif ontology.is_relation(item['name']):
                if not ontology.is_relation(item['name'],mainEntity):
                    slots[i]={}
            elif item['name'].startswith('${#'):
                slots[i]={}
        slots=[i for i in slots if i!={}]
        semantic['slots']=slots
        semantic['mainEntity']=''
    elif sentence_tag=='nil':
        for i,item in enumerate(slots):
            if ontology.is_attribute(item['name']):
                slots[i]={} #直接去掉
            elif ontology.is_relation(item['name']): # relation 变 entity
                item['name']='${#'+ontology.get_type(item['name'])+'}'
                slots[i]=item
        slots=[i for i in slots if i!={}]
        semantic['slots']=slots
        semantic['mainEntity']=slots[0]['value'] if len(slots)>0 else '' #选第一个作为mainEntity
    elif ontology.is_attribute(sentence_tag):
        #只需要找出mainEntity,slots只有一个为mainEntity
        mainEntity_list=ontology.get_entity_list(sentence_tag)
        first_candidate,second_candidate,third_candidate=[],[],[]
        for i,item in enumerate(slots):
            name=item['name']
            if name.startswith('${#') and name.strip('${#}') in mainEntity_list:
                first_candidate.append(item)
            elif ontology.is_relation(name) and ontology.get_type(name) in mainEntity_list:
                second_candidate.append(item)
            elif name.startswith('${#') or ontology.is_relation(name):
                third_candidate.append(item)
        if first_candidate!=[]: #启发式：如果有多个挑选第一个,随机挑还是用更intelligent的方法
            semantic['mainEntity']=first_candidate[0]['value']
            semantic['slots']=[first_candidate[0]]
        elif second_candidate!=[]:            
            semantic['mainEntity']=second_candidate[0]['value']
            second_candidate[0]['name']='${#'+ontology.get_type(second_candidate[0]['name'])+'}'
            semantic['slots']=[second_candidate[0]]
        elif third_candidate!=[]:
            semantic['mainEntity']=third_candidate[0]['value']
            third_candidate[0]['name']=random.choice(mainEntity_list)
            semantic['slots']=[third_candidate[0]]
        else:#如果根本找不到mainEntity,return type改为nil，解析结果为空
            semantic['mainEntity']=''
            semantic['return_type']='nil'
            semantic['slots']=[]
    else:
        # 先挑选mainEntity
        linked_entity=ontology.get_type(sentence_tag)
        mainEntity_list=ontology.get_entity_list(sentence_tag)
        first_candidate,second_candidate,third_candidate=[],[],[]
        for i,item in enumerate(slots):
            name=item['name']
            if name.startswith('${#') and name.strip('${#}') in mainEntity_list:
                first_candidate.append(item)
            elif ontology.is_relation(name) and ontology.get_type(name) in mainEntity_list:
                second_candidate.append(item)
            elif name.startswith('${#') or ontology.is_relation(name):
                third_candidate.append(item)
        if first_candidate!=[]: #启发式：如果有多个挑选第一个,随机挑还是用更intelligent的方法
            semantic['mainEntity']=first_candidate[0]['value']
            mainEntity,choice=first_candidate[0],1
        elif second_candidate!=[]:            
            semantic['mainEntity']=second_candidate[0]['value']
            mainEntity,choice=second_candidate[0],2
        elif third_candidate!=[]:
            semantic['mainEntity']=third_candidate[0]['value']
            mainEntity,choice=third_candidate[0],3
        else:#如果根本找不到mainEntity,return type改为nil，解析结果为空
            semantic['mainEntity']=''
            semantic['return_type']='nil'
            semantic['slots']=[]
        if semantic['mainEntity']!='':
            for i,item in enumerate(slots):
                # 再次遍历，确认每个slot
                name=item['name']
                if item==mainEntity:
                    if choice==1:
                        pass
                    elif choice==2:
                        item['name']=ontology.get_type(item['name'])
                        slots[i]=item
                    else:
                        item['name']=random.choice(mainEntity_list)
                        slots[i]=item
                elif not (ontology.is_attribute(item['name'],linked_entity) or ontology.is_relation(item['name'],linked_entity)):
                    slots[i]={}
                else:
                    pass
            slots=[i for i in slots if i!={}]
            semantic['slots']=slots
    return json.loads(semantic,ensure_ascii=False) if str_flag else semantic

def encapsulate_semantics(string_dict,ontology):
    # 将{"return_type": "description", "mainEntity": "实验室", "slots": [{"op": "eq", "name": "${#institute}", "value": "实验室"}]}
    # 转化为
    # {"ne_0":{"value":"实验室","type":"insitute"},"er_1":{"pred":"description","filter":[],"ent_idx":"ne_0"}}
    if type(string_dict)==str:
        string_dict=json.loads(string_dict.strip())
    ent_info={}
    pred_info={}
    if string_dict['return_type'].startswith('${#'):
        pred_info['er_1']={}
        if "mainEntity" in string_dict:
            assert string_dict["mainEntity"]==""
        pred_info["er_1"]['expr_type']="entity"
        pred_info["er_1"]['pred']=string_dict['return_type'].strip('{}#$')
        pred_info["er_1"]["filter"]=[]
        pred_info["er_1"]['ent_idx']=None
        idx=0
        for each in string_dict['slots']:
            name=each['name'].strip('{}#$')
            value=each['value']
            if ontology.is_relation(name,pred_info["er_1"]['pred']):
                ent_info["ne_"+str(idx)]={"value":value,"type":ontology.get_type(name,pred_info["er_1"]['pred'])}
                pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":name,"value":"ne_"+str(idx)})
                idx+=1
            elif ontology.is_attribute(name,pred_info["er_1"]['pred']):
                pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":name,"value":value})
            elif name in ontology.get_rel_type_list(pred_info["er_1"]['pred']):
                rel_name_list=ontology.get_rel_name_list(pred_info["er_1"]['pred'],rel_type=name)
                if len(rel_name_list)>0:
                    constraint=rel_name_list[0]
                    ent_info["ne_"+str(idx)]={"value":value,"type":name}
                    pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":constraint,"value":"ne_"+str(idx)})
                    idx+=1
            elif ontology.is_relation(name) and ontology.get_type(name) in ontology.get_rel_type_list(pred_info["er_1"]['pred']):
                rel_name_list=ontology.get_rel_name_list(pred_info["er_1"]['pred'],rel_type=ontology.get_type(name))
                if len(rel_name_list)>0:
                    constraint=rel_name_list[0]
                    ent_info["ne_"+str(idx)]={"value":value,"type":ontology.get_type(name)}
                    pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":constraint,"value":"ne_"+str(idx)})
                    idx+=1
        pred_info['er_'+str(idx)]=pred_info.pop('er_1')
        return dict(ent_info,**pred_info)
    elif string_dict['return_type']=="nil":
        idx=0
        for each in string_dict['slots']:
            value=each['value']
            type_=each['name'].strip('{}#$')
            if ontology.is_entity(type_):
                ent_info["ne_"+str(idx)]={"value":value,"type":type_}
                idx+=1
            elif ontology.is_relation(type_):
                ent_info['ne_'+str(idx)]={'value':value,"type":ontology.get_type(type_)}
                idx+=1
        return ent_info #只有entity，没有pred，如果连slot都没有，就返回空字典
    else:
        # return type=rel or attr
        pred_info['er_1']={}
        pred_info['er_1']['expr_type']='rel_attr'
        pred_info["er_1"]['pred']=string_dict['return_type']
        pred_info["er_1"]["filter"]=[]
        pred_info["er_1"]['ent_idx']=""
        link_type=ontology.get_type(pred_info["er_1"]['pred'])
        idx=0
        for each in string_dict['slots']:
            name=each['name']
            value=each['value']
            if name.startswith('${#'): #mainEntity
                name=name.strip('{}#$')
                if (string_dict['mainEntity']=='' or (string_dict['mainEntity']!='' and value==string_dict['mainEntity'])) \
                and pred_info["er_1"]['ent_idx']=="" and name in ontology.get_entity_list(pred_info["er_1"]['pred']):
                    ent_info["ne_"+str(idx)]={"value":value,"type":name}
                    pred_info["er_1"]["ent_idx"]="ne_"+str(idx)
                    idx+=1
                elif link_type not in ['string','int','float'] and name in ontology.get_rel_type_list(link_type):
                    rel_name_list=ontology.get_rel_name_list(link_type,rel_type=name)
                    if len(rel_name_list)>0:
                        constraint=rel_name_list[0]
                        ent_info["ne_"+str(idx)]={"value":value,"type":name}
                        pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":constraint,"value":"ne_"+str(idx)})
                        idx+=1
            elif link_type not in ['string','int','float']: # rel, not attr
                if ontology.is_relation(name,link_type):
                    ent_info["ne_"+str(idx)]={"value":value,"type":ontology.get_type(name,link_type)}
                    pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":name,"value":"ne_"+str(idx)})
                    idx+=1
                elif ontology.is_attribute(name,link_type):
                    pred_info["er_1"]["filter"].append({"op":each["op"],"constraint":name,"value":value})
        if pred_info['er_1']['ent_idx']!='':
            pred_info['er_'+str(idx)]=pred_info.pop('er_1')
            return dict(ent_info,**pred_info)
        else: #没有找到mainEntity
            return ent_info

if __name__=='__main__':

    ontology=PATH_TO_ONTOLOGY['SpeechLab']
    json_ontology=Ontology(ontology)

    annotation_file=os.path.join(os.path.dirname(PATH_TO_TEST_DATA['SpeechLab']),'test.annotation.txt')
    with open(annotation_file,'r') as inf:
        for line in inf:
            line=line.strip()
            if line=='':
                continue
            line=line.split('=>')[2].strip()
            print(encapsulate_semantics(line,json_ontology))

    # string='{"return_type": "papers", "mainEntity": "你们实验室", "slots": [{"op": "eq", "name": "${#institute}", "value": "你们实验室"}, {"op": "eq", "name": "publication_date", "value": "两010年"}, {"op": "eq", "name": "published_in", "value": "apsipa"}]}'
    # print(encapsulate_semantics(string,json_ontology))
