#!/usr/bin/env python3
#coding=utf8

import os,sys,argparse,random,re
from collections import defaultdict

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from global_vars import *
from utils import preproc

def build_trainset_and_voc_char_level(ifilename,ofilename,vocabulary_file,shuffle=True):
    with open(ifilename,'r') as infile,open(ofilename,'w') as outfile,open(vocabulary_file,'w') as vocfile:
        vocabulary=defaultdict(lambda :0)
        trainingset=[]
        for eachline in infile:
            item=[]
            splits=eachline.split('=>')
            rule=splits[0].strip()
            node_info=splits[1].strip()
            semantics=splits[2].strip()

            item.append(node_info)
            item.append(semantics)

            char_list=[]
            bio_list=[]
            chunk_list=re.split(r'(\(.*?\)|\[.*?\])',rule)
            for each_chunk in chunk_list:
                if each_chunk.startswith('[') or each_chunk.startswith('('):
                    each_chunk=each_chunk.strip('[]()')
                    name,tag=each_chunk.split(':')
                    tmp_char_list=[i for i in preproc(name).split(' ') if i!='']
                    tmp_bio_list=[]
                    tag=tag.split('=')[1]
                    for each_char in tmp_char_list:
                        vocabulary[each_char]+=1
                        tmp_bio_list.append('I-'+tag)
                    char_list.extend(tmp_char_list)
                    tmp_bio_list[0]='B-'+tag
                    bio_list.extend(tmp_bio_list)
                else:
                    tmp_char_list=[i for i in preproc(each_chunk).split(' ') if i!='']
                    if tmp_char_list==[]:
                        continue
                    tmp_bio_list=[]
                    for each_char in tmp_char_list:
                        vocabulary[each_char]+=1
                        tmp_bio_list.append('O')
                    char_list.extend(tmp_char_list)
                    bio_list.extend(tmp_bio_list)
            item.append((char_list,bio_list))
            trainingset.append(item)

        if shuffle:
            random.shuffle(trainingset)

        for each in trainingset:
            outfile.write('\n') # 用来区分每一个sample
            outfile.write(each[0]+'\n') # 第一个位置是sentence classification结果
            outfile.write(each[1]+'\n') # 第二个位置是解析的语义槽
            char_list,bio_list=each[2]
            assert len(char_list)==len(bio_list)
            for idx,_ in enumerate(bio_list):
                outfile.write(char_list[idx]+'\t'+bio_list[idx]+'\n')
        total_num=len(trainingset)
        vocfile.write('<PAD>\t0\t0\n')
        vocfile.write('<UNK>\t1\t0\n')
        index=2
        tuples=list(sorted(vocabulary.items(),key=lambda x:x[1],reverse=True))
        for voc,count in tuples:
            vocfile.write(voc+'\t'+str(index)+'\t'+str(count)+'\n')
            index+=1

def build_trainset_and_voc_word_level(infile,outfile,vocfile):
    pass

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--domain',default='SpeechLab')
    args=parser.parse_args()
    infile=os.path.join(PATH_TO_DATA[args.domain],'rules.annotation.txt')
    outfile=PATH_TO_TRAINING_DATA[args.domain]
    vocabulary_file=PATH_TO_VOC[args.domain]
    build_trainset_and_voc_char_level(infile,outfile,vocabulary_file)

    # dataset format: first line is sentence classification result,second line is json format semantics
    # description
    # {"return_type": "description", "mainEntity": "语义解析类", "slots": [{"op": "eq", "name": "${#concept}", "value": "语义解析类"}]}
    # 语   B-${#concept}
    # 义   I-${#concept}
    # 解   I-${#concept}
    # 析   I-${#concept}
    # 类   I-${#concept}
    # 主   O
    # 要   O
    # 指   O
    # 的   O
    # 是   O
    # 什   O
    # 么   O
    # 意   O
    # 思   O