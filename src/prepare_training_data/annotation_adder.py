#!/usr/bin/env python3
#coding=utf8
'''
@Time   : 2018-03-11
@Author : ruisheng.cao
@Desc   : replace entities and constraints in rules.release.txt
@used   : python annotation_adder.py -d domain
'''

import fst
import argparse
import os,sys,re,json
import random,copy
from collections import defaultdict

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from global_vars import *
from utils import reverse_preproc

class Annotation_adder():
    '''
    [description] 
        根据tagger和constraint的fst对rules.release.txt里的实体和限制进行替换并添加语义解析标注
    '''
    def __init__(self,dir_to_tagger,dir_to_phrase,infile,outfile):

        self._prepare_resource(dir_to_tagger,dir_to_phrase)
        self.infile=open(infile,'r') if type(infile)==str else infile
        self.outfile=open(outfile,'w') if type(outfile)==str else outfile

    def _prepare_resource(self,dir_to_tagger,dir_to_phrase):
        '''
        [description]
            根据tagger和constraint的fst生成字典
        Arguments:
            dir_to_tagger {string} -- [description]
            dir_to_phrase {string} -- [description]
        Returns:
            tagger_dict -- [description] tagger_dict['${.concept}']=string list, each string is a path
            constraint_dict -- [description] constraint_dict['${@constraint}']=list of (string path, mapped value)
        '''
        # deal with entities(tagger)
        files=os.listdir(dir_to_tagger)
        isyms=fst.read_symbols(os.path.join(dir_to_tagger,'isyms.fst'))
        osyms=fst.read_symbols(os.path.join(dir_to_tagger,'osyms.fst'))
        filepath=os.path.join(dir_to_tagger,[each for each in files if each not in ['isyms.fst','osyms.fst'] and each.endswith('.fst')][0])
        lexicon=fst.read_std(filepath)
        lexicon.isyms=isyms
        lexicon.osyms=osyms
        self.tagger_dict=defaultdict(list)
        for each_path in lexicon.paths():
            input_string=[lexicon.isyms.find(arc.ilabel) for arc in each_path if arc.ilabel != 0]
            if len(input_string)!=1:
                raise ValueError('[Error]:error in resolving tagger name!')
            output_string=[lexicon.osyms.find(arc.olabel) for arc in each_path if arc.olabel != 0]
            self.tagger_dict[input_string[0]].append(reverse_preproc(output_string))
        # deal with constraints
        files=os.listdir(dir_to_phrase)
        isyms=fst.read_symbols(os.path.join(dir_to_phrase,'isyms.fst'))
        osyms=fst.read_symbols(os.path.join(dir_to_phrase,'osyms.fst'))
        fst_dict={}
        for each in files:
            if each not in ['isyms.fst','osyms.fst'] and each.endswith('.fst'):
                fst_dict[each[0]]=fst.read_std(os.path.join(dir_to_phrase,each))
                fst_dict[each[0]].isyms=isyms
                fst_dict[each[0]].osyms=osyms
        self.constraint_dict=defaultdict(list)
        for each in sorted(fst_dict.keys()): #层级phrase的fst按0-1-2-...顺序组织
            tmp_fst=fst_dict[each]
            for path in tmp_fst.paths():
                name,item_list=self._get_path_and_mapped_value(path,tmp_fst)
                self.constraint_dict[name].extend(item_list)
        return (self.tagger_dict,self.constraint_dict)

    def _get_path_and_mapped_value(self,path,tmp_fst):
        '''
        [description]
            Get the constraint name , all paths and their mapped value from a single path,
            recursively extend the phrases ${xxx} in the path
        Arguments:
            path {fst.path} -- [description] path in a fst
            tmp_fst {fst} -- [description] fst where the path is from
        Returns:
            (constraint_name,[(string path,mapped value),... ...])
        '''
        istring=[tmp_fst.isyms.find(arc.ilabel) for arc in path if arc.ilabel!=0]
        ostring=[tmp_fst.osyms.find(arc.olabel) for arc in path if arc.olabel!=0]
        name=istring[0]
        if len(istring[1:])==len(ostring)+1:
            mapped_value=istring[1]
            merge=zip(istring[2:],ostring)
            return_list=[list()]
            for pair in merge:
                if pair[1].startswith('${') and (not pair[1].startswith('${.')):
                    if pair[1] not in self.constraint_dict:
                        raise ValueError('[Error]: not found phrase',pair[1],'while dealing with',name)
                    extensions=self.constraint_dict[pair[1]]
                    old_return_list=copy.deepcopy(return_list)
                    return_list=[]
                    for each in old_return_list:
                        for substring in extensions:
                            item=substring[0]
                            return_list.append(copy.deepcopy(each).append(item))
                else:
                    for each in return_list:
                        each.append(pair[1])
            item_list=[]
            for each in return_list:
                item_list.append((reverse_preproc(each),mapped_value))
            return (name,item_list)
        else:
            if len(istring[1:])!=len(ostring):
                raise ValueError('[Error]:error in input and output labels of phrase',name)
            merge=zip(istring[1:],ostring)
            return_list=[[list(),list()]]
            for pair in merge:
                if pair[0]=='_':
                    if pair[1].startswith('${') and not pair[1].startswith('${.'):
                        if pair[1] not in self.constraint_dict:
                            raise ValueError('[Error]: not found phrase',pair[1],'while dealing with',name)
                        extensions=self.constraint_dict[pair[1]]
                        old_return_list=copy.deepcopy(return_list)
                        return_list=[]
                        for each in old_return_list:
                            for substring in extensions:
                                current=copy.deepcopy(each)
                                current[0].append(substring[0])
                                return_list.append(current)
                    else:
                        for each in return_list:
                            each[0].append(pair[1])
                elif pair[0]=='$':
                    if pair[1].startswith('${') and not pair[1].startswith('${.'):
                        if pair[1] not in self.constraint_dict:
                            raise ValueError('[Error]: not found phrase',pair[1],'while dealing with',name)
                        extensions=self.constraint_dict[pair[1]]
                        old_return_list=copy.deepcopy(return_list)
                        return_list=[]
                        for each in old_return_list:
                            for substring in extensions:
                                current=copy.deepcopy(each)
                                current[0].append(substring[0])
                                current[1].append(substring[1])
                                return_list.append(current)
                    else:
                        for each in return_list:
                            each[0].append(pair[1])
                            each[1].append(pair[1])
                else:
                    raise ValueError('[Error]:unrecognized inlabel',pair[0],'in constraint',name)
            item_list=[]
            for each in return_list:
                item_list.append((reverse_preproc(each[0]),reverse_preproc(each[1])))
            return (name,item_list)

    def fill_slots_and_generate_tagset(self):
        '''        
        [description]
            fill in entities and constraints in rules.release.txt and generate tagset
        Returns:
            sentence_tagset [type] -- [description]
        '''
        infile=self.infile
        outfile=self.outfile
        sentence_tagset=set()
        bio_tagset=set()
        bio_tagset.add('O') 
        sentence_pattern=re.compile(r'\(\s*\((.*?)\)\s*,\s*\[.*?\]\);?')
        filter_pattern=re.compile(r'\(\s*\(.*?\)\s*,\s*\[\s*(.*?)\s*\]\);?')
        for eachline in infile:
            rule,node=eachline.split('=>')
            rule=rule.strip()
            node=node.strip()
            sentence_tag=sentence_pattern.findall(node)[0]
            filters=filter_pattern.findall(node)[0]
            semantics={'return_type':'','mainEntity':'','slots':list()}

            if ',' in sentence_tag:
                # return relation or attribute
                return_type=sentence_tag.split(',')[1]
                main_entity_idx=int(sentence_tag.split(',')[0].strip(' #'))
                main_entity_flag=True            
            else:
                # return entity
                if sentence_tag!='nil':
                    return_type='${#'+sentence_tag+'}'
                    main_entity_flag=False
                else:
                    return_type='nil'
                    main_entity_flag=True
            sentence_tagset.add(return_type)
            semantics['return_type']=return_type

            if filters=='':
                filters={}
            else:
                filter_list=re.findall(r'\((.*?)\)',filters)
                filters=dict()
                for each in filter_list:
                    op=each.split(',')[0].strip()
                    tag=each.split(',')[1].strip()
                    tag_idx=each.split(',')[2].strip()
                    if tag_idx in filters:
                        print('[Warning]:duplicate use of constraints',tag_idx,'in',eachline)
                    filters[tag_idx]=(op,tag)
            count,begin=0,0
            while begin<len(rule) and '${#' in rule[begin:]:
                start_idx=rule[begin:].find('${#')
                end_idx=start_idx+3
                count+=1
                while rule[begin:][end_idx]!='}':
                    end_idx+=1
                replace=rule[begin:][start_idx:end_idx+1]
                replace_with=random.choice(self.tagger_dict['${.'+replace[3:]])
                if main_entity_flag and (return_type=='nil' or int(count)==int(main_entity_idx)):
                    rule=rule[:begin]+rule[begin:][:start_idx]+'('+replace_with+':mainEntity='+replace+')'+rule[begin:][end_idx+1:]
                    bio_tagset.add('B-'+replace)
                    bio_tagset.add('I-'+replace)
                    begin=begin+start_idx+len('('+replace_with+':mainEntity='+replace+')')
                    semantics['mainEntity']=replace_with
                    semantics['slots'].append({'op':'eq','name':replace,'value':replace_with})
                elif '#'+str(count) in filters:
                    rule=rule[:begin]+rule[begin:][:start_idx]+'['+replace_with+':'+replace+'='+filters['#'+str(count)][1]+']'+rule[begin:][end_idx+1:]
                    bio_tagset.add('B-'+filters['#'+str(count)][1])
                    bio_tagset.add('I-'+filters['#'+str(count)][1])
                    begin=begin+start_idx+len('['+replace_with+':'+replace+'='+filters['#'+str(count)][1]+']')
                    semantics['slots'].append({'op':filters['#'+str(count)][0],'name':filters['#'+str(count)][1],'value':replace_with})
                else:
                    print('[Warning]:unused entity',replace,'in',rule)
                    rule=rule[:begin]+rule[begin:][:start_idx]+replace_with+rule[begin:][end_idx+1:]
                    begin=begin+start_idx+len(replace_with)

            count=0
            begin=0
            while begin<len(rule) and '${@' in rule[begin:]:
                start_idx=rule[begin:].find('${@')
                end_idx=start_idx+3
                count+=1
                while rule[begin:][end_idx]!='}':
                    end_idx+=1
                replace=rule[begin:][start_idx:end_idx+1]
                replace_with=random.choice(self.constraint_dict[replace])[0]
                if '@'+str(count) in filters:
                    rule=rule[:begin]+rule[begin:][:start_idx]+'['+replace_with+':'+replace+'='+filters['@'+str(count)][1]+']'+rule[begin:][end_idx+1:]
                    bio_tagset.add('B-'+filters['@'+str(count)][1])
                    bio_tagset.add('I-'+filters['@'+str(count)][1])
                    begin=begin+start_idx+len('['+replace_with+':'+replace+'='+filters['@'+str(count)][1]+']')
                    semantics['slots'].append({'op':filters['@'+str(count)][0],'name':filters['@'+str(count)][1],'value':replace_with})
                else:
                    print('[Warning]:unused constraint',replace,'in',rule)
                    rule=rule[:begin]+rule[begin:][:start_idx]+replace_with+rule[begin:][end_idx+1:]
                    begin=begin+start_idx+len(replace_with)

            outfile.write(rule+' => '+return_type+' => '+json.dumps(semantics,ensure_ascii=False)+'\n')

        return sentence_tagset,bio_tagset

    def get_constraint_mappings(self,output):
        '''
        [description]
            Get the mappings from constraints to their mapped values and write into file        
        Arguments:
            output {string} -- [description] string indicates where to store the file
        '''
        output=open(output,'w') if type(output)==str else output
        mappings_set=set()
        for each in self.constraint_dict:
            item_list=self.constraint_dict[each]
            for each_item in item_list:
                mappings_set.add('=>'.join([each_item[0],each_item[1]])+'\n')
        for each in mappings_set:
            output.write(each)
            
if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--domain',default='AISpeech',help='which domain to deal with')
    parser.add_argument('--test',action='store_true')
    args=parser.parse_args()
    if args.test:
        file=os.path.join(PATH_TO_DATA[args.domain],'rules.test.release.txt')
        out=os.path.join(PATH_TO_DATA[args.domain],'rules.test.annotation.txt')
    else:
        file=os.path.join(PATH_TO_DATA[args.domain],'rules.release.txt')
        out=os.path.join(PATH_TO_DATA[args.domain],'rules.annotation.txt')

    annotation_adder=Annotation_adder(os.path.join(PATH_TO_FST[args.domain],'lexicon/'),
        os.path.join(PATH_TO_FST[args.domain],'constraint/'),file,out)
    
    annotation_adder.get_constraint_mappings(os.path.join(PATH_TO_DATA[args.domain],'constraint_mappings.txt'))
    
    sentence_tagset,slot_tagset=annotation_adder.fill_slots_and_generate_tagset() 
    
    if not args.test:
        sentence_tagset=list(sorted(sentence_tagset))
        slot_tagset=list(sorted(slot_tagset))
        with open(PATH_TO_SENTENCE_TAGSET[args.domain],'w') as out1,open(PATH_TO_SLOT_TAGSET[args.domain],'w') as out2:
            for i,each in enumerate(sentence_tagset):
                out1.write(each+'\t'+str(i)+'\n')
            for i,each in enumerate(slot_tagset):
                out2.write(each+'\t'+str(i)+'\n')
            out2.write()

