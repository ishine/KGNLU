#!/usr/bin/env python3
#coding=utf8
'''
@Time   : 2018-03-09
@Author : ruisheng.cao
@Desc   : It is used to extract all paths of a limited regular experssion (regex). The limited regex supports the special characters:
            1. ()
            2. ?
            3. |
@used   : python -d domain -w weight
'''
import os,sys
import argparse,json,re,random
import fst
from pyltp import Segmentor

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from global_vars import *

concept_fst_dict = {}
constraints_names = {}

def main(argv):
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d','--domain',default='AISpeech',action='store',help='which domain: AISpeech or SpeechLab')
    parser.add_argument('-w','--weight',default=-1,action='store',metavar='number',type=float,help='weight number')
    parser.add_argument('--test',action='store_true')
    args = parser.parse_args()
    lex_file = open(os.path.join(PATH_TO_DATA[args.domain],'rules.txt'), 'r')
    weight = args.weight
    if not args.test:
        out_lex_file = open(os.path.join(PATH_TO_DATA[args.domain],'rules.release.txt'), 'w')
    else:
        out_lex_file = open(os.path.join(PATH_TO_DATA[args.domain],'rules.test.release.txt'), 'w')
    cws_model_path = PATH_TO_SPLIT_WORDS  # 分词模型路径，模型名称为`cws.model`
    dict_path = os.path.join(PATH_TO_DATA[args.domain], 'dict.txt') # 领域相关的词典，用于帮助分词
    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon(cws_model_path,dict_path) # 加载模型

    if concept_fst_dict!={}:
        concept_fst_dict.clear()
    if constraints_names!={}:
        constraints_names.clear()

    macro_patterns = {}
    all_patterns = []
    for line in lex_file:
        line=line.strip()
        if line == '' or line.startswith('%'):
            continue
        if '=>' not in line:
            #规则宏
            pat_name, pat = line.strip(';').split('=')
            macro_patterns['${'+pat_name+'}'] = extract_simple_rules(pat.strip(), macro_patterns)
        else:
            #正常规则
            pattern, node_info = line.split('=>')
            chunk_list = extract_simple_rules(pattern.strip(), macro_patterns)
            all_patterns.append((chunk_list, node_info))

    isyms = ["<eps>"]
    label_voc = {}
    osyms = ["<eps>", "<unk>"]
    word_voc = {} #["<unk>"] #<unk> should be defined manually
    for chunk_list,_ in all_patterns:
        for word in chunk_list:
            if word[0] not in ['(', ')', '|']:
                word = word.strip('?')
                word_voc[word] = 1

    osyms = osyms + list(word_voc)
    osyms_table = fst.SymbolTable()
    for idx,val in enumerate(osyms):
        osyms_table[val] = idx
    
    isyms_table = fst.SymbolTable()
    for idx,val in enumerate(isyms):
        isyms_table[val] = idx

    for pattern_idx, (pattern_chunk_list, node_info) in enumerate(all_patterns):
        # unique_rules = set()
        replace_mapping_dict = {}
        concept_fst = fst.StdTransducer(isyms=isyms_table, osyms=osyms_table)
        segment_stack = [{'start_of_this_segment':0, 'end_of_this_segment':0}]
        segment_stack[0]['value'] = '<eps>'
        cursor_head, cursor_tail = 0, 1
        argument_count = 0
        # print('Processing rule',pattern_chunk_list,'=>',node_info)
        for word in pattern_chunk_list:
            if word == '(':
                argument_count += 1
                segment_stack.append({'start_of_this_segment':cursor_tail, 'end_of_this_segment':0, 'value':segment_stack[-1]['value']})
                segment_stack[-1]['head_arc'] = [cursor_head, cursor_tail]
                cursor_tail += 1
                cursor_head = cursor_tail - 1
            elif word[0] == ')':
                if segment_stack[-1]['end_of_this_segment'] == 0:
                    segment_stack[-1]['end_of_this_segment'] = cursor_head
                else:
                    concept_fst.add_arc(cursor_head, segment_stack[-1]['end_of_this_segment'], '<eps>', '<eps>')
                    cursor_head = segment_stack[-1]['end_of_this_segment']
                if word == ')?':
                    concept_fst.add_arc(segment_stack[-1]['head_arc'][0], segment_stack[-1]['head_arc'][1], '<eps>', '<eps>')
                    concept_fst.add_arc(segment_stack[-1]['start_of_this_segment'], segment_stack[-1]['end_of_this_segment'], '<eps>', '<eps>')
                else:
                    concept_fst.add_arc(segment_stack[-1]['head_arc'][0], segment_stack[-1]['head_arc'][1], '<eps>', '<eps>')
                segment_stack.pop()
            elif word == '|':
                if segment_stack[-1]['end_of_this_segment'] == 0:
                    segment_stack[-1]['end_of_this_segment'] = cursor_head
                else:
                    concept_fst.add_arc(cursor_head, segment_stack[-1]['end_of_this_segment'], '<eps>', '<eps>')
                cursor_head = segment_stack[-1]['start_of_this_segment']
            else:
                if word[-1] == '?':
                    concept_fst.add_arc(cursor_head, cursor_tail, '<eps>', '<eps>')
                    word = word[:-1]
                else:
                    pass
                next_state = add_arc(concept_fst, cursor_head, cursor_tail, word, segment_stack[-1]['value'])
                cursor_head = cursor_tail
                cursor_tail = next_state
        if segment_stack[-1]['end_of_this_segment'] == 0:
            segment_stack[-1]['end_of_this_segment'] = cursor_head
        else:
            concept_fst.add_arc(cursor_head, segment_stack[-1]['end_of_this_segment'], '<eps>', '<eps>')
        final_state_idx = segment_stack[-1]['end_of_this_segment']
        concept_fst[final_state_idx].final = True
        
        concept_fst = concept_fst.inverse()
        concept_fst = concept_fst.determinize()
        concept_fst.minimize()
        concept_fst = concept_fst.inverse()
        
        t = concept_fst
        paths=list(t.paths())
        random.shuffle(paths)
        if not args.test:
            if extract_proper_num(len(paths))>len(paths):
                paths=paths*(extract_proper_num(len(paths))//len(paths))+paths[:extract_proper_num(len(paths))%len(paths)]
            else:
                paths=paths[:extract_proper_num(len(paths))]
        else:
            paths=paths[:2] if len(paths)>=2 else paths
        for output in paths:
            raw_path = []
            for arc in output:
                raw_path.append((t.osyms.find(arc.olabel), t.isyms.find(arc.ilabel)))
            path = raw_path
            input_seq = []
            output_seq = []
            for word, label in path:
                if word not in ['<eps>', u"ε"]:
                    input_seq.append(word)
                if label not in ['<eps>', u"ε"]:
                    if label == '_' and word not in ['<eps>', u"ε"]:
                        output_seq.append(word)
                    elif label != '_':
                        output_seq.append(label)
            
            pattern = input_seq
            sentence = [item if item[0] != '$' else ',' for item in pattern]
            tags = [item for item in pattern if item[0] == '$']
            sentence = ''.join(sentence)
            words = segmentor.segment(sentence)

            new_words = []
            tag_idx = 0
            for word in words:
                word = word
                if word == ',':
                    word = tags[tag_idx]
                    tag_idx += 1
                new_words.append(word)

            new_rule_simple = ' '.join(new_words)+' => '+node_info
            out_lex_file.write(new_rule_simple+'\n')
            # if new_rule_simple not in unique_rules:
            #     out_lex_file.write(new_rule_simple+'\n')
            # unique_rules.add(new_rule_simple)

def add_arc(my_fst, fromstate, tostate, word, label):
    next_state = tostate+1
    my_fst.add_arc(fromstate, tostate, label, word)
    return next_state

def extract_simple_rules(pattern, macro_patterns):
    pattern = re.sub('[ ]+', ' ', pattern)
    pattern_chunk_list = []
    max_length = len(pattern)
    idx = 0
    while idx < max_length:
        cur_word = pattern[idx]
        if cur_word == ')':
            if idx+1 < max_length and pattern[idx+1] in ['?']:
                pattern_chunk_list.append(pattern[idx:idx+2])
                idx += 2
            else:
                pattern_chunk_list.append(cur_word)
                idx += 1
        elif cur_word == '$':
            if idx+1 == max_length:
                raise ValueError("[Error]: raw pattern error!")
            elif pattern[idx+1] != '{':
                raise ValueError("[Error]: raw pattern error!")
            else:
                end_idx = idx+2
                while pattern[end_idx] != '}':
                    end_idx += 1
                if pattern[idx:end_idx+1][0:3] in ('${#', '${@'):
                    if end_idx+1 < max_length and pattern[end_idx+1] in ['?']: #['*', '+', '?']:
                        pattern_chunk_list.append(''.join(pattern[idx:end_idx+2]))
                        idx = end_idx+2
                    else:
                        pattern_chunk_list.append(''.join(pattern[idx:end_idx+1]))
                        idx = end_idx+1
                else:
                    if pattern[idx:end_idx+1] not in macro_patterns:
                        print(pattern[idx:end_idx+1])
                        raise ValueError("[Error]: macro pattern is missing!")
                    if end_idx+1 < max_length and pattern[end_idx+1] in ['?']: #['*', '+', '?']:
                        pattern_chunk_list += ['('] + macro_patterns[pattern[idx:end_idx+1]] + [')'+pattern[end_idx+1]]
                        idx = end_idx+2
                    else:
                        pattern_chunk_list += ['('] + macro_patterns[pattern[idx:end_idx+1]] + [')']
                        idx = end_idx+1
        elif cur_word == ' ':
            idx += 1
        else:
            if idx+1 < max_length and pattern[idx+1] in ['?']:
                pattern_chunk_list.append(pattern[idx:idx+2])
                idx += 2
            else:
                pattern_chunk_list.append(cur_word)
                idx += 1
    return pattern_chunk_list
    
def extract_proper_num(num):
    if num==None or (not isinstance(num,int)):
        return 0
    elif num<=0:
        return 0
    elif num>=5000:
        return 5000
    elif num<1000:
        return 1000
    else:
        return num

if __name__ == '__main__':
    
    main(sys.argv) 
