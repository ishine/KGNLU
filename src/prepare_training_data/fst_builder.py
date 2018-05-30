#!/usr/bin/env python
'''
@Time   : 2018-03-22 19:54:49
@Author : ruisheng.cao
@Desc   : Build fst for lexicons and constraints
'''

import os,sys,time
import argparse,threading
import fst

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
import utils
from global_vars import *

class PhraseFSTBuilder():
    r"""
    从短语规则主文件中读取所有的短语规则，对短语规则之间的引用关系进行拓扑分层排序；对每一个短语规则分别单独构建FST，然后按“拓扑层”依次合并为一个个整体的短语匹配FST。

    已经支持：
        1. 引用: phrase_name=(${pre_rule_name_1}|${pre_rule_name_2})xxx;
        2. 引用DB: phrase_name=(${.person}|${pre_rule_name_2})xxx;
        3. value mapping: phrase_name=(国内|我国) => ("中国");
        4. value mapping: phrase_name=(语音|语言)类 => ("$1");  #此处 $N 按左括号'('进行顺序编号

    Arguments:
        main_lex (path): 短语规则主文件路径；
        fst_path (dir): FST存储路径。
        weight (float): FST边权重；

        FSTs (dict): 该字典包含每一条短语LEX的FST；
    """
    def __init__(self, main_lex, fst_path, weight=-1):
        self.phrase_dict, self.phrase_names_in_order = self._read_data(main_lex)
        self.weight = weight
        self.FSTs = {}

        if not os.path.isdir(fst_path):
            os.makedirs(fst_path)
        else:
            for filename in os.listdir(fst_path):
                targetFile = os.path.join(fst_path,filename)
                if os.path.isfile(targetFile):
                    os.remove(targetFile)
        self.fst_path = fst_path

        self.isyms, self.osyms = self._build_vocab()
    
    def _read_data(self, main_lex):
        r"""
        从规则主文件读取所有的短语规则。
            1. 读取所有短语规则；
            2. 处理正则文本，自动分段
            3. 根据短语规则之间的引用关系进行拓扑排序和分层；

        Arguments:
            main_lex (path): 短语规则主文件路径；
        Returns:
            phrase_dict (dict)：短语规则数据；
            phrase_names_in_order (list): 拓扑排序后的规则名称，比如该list第一个元素表示的是所有没有依赖的短语规则；
        """
        # 1. 读取所有短语规则；
        contents = self._analysis_lex(main_lex)
        # 2. 处理正则文本，自动分段
        phrase_dict = {}
        phrase_names_topology = {}
        for line in contents:
            items = line.split('=')
            phrase_name, pattern = items[0], '='.join(items[1:])
            if '=>' in pattern:
                pattern, mapped_value = pattern.split('=>')
                pattern = pattern.strip(' ')
                mapped_value = mapped_value.strip(' ')
            else:
                mapped_value = ''
            used_name = '${'+phrase_name+'}'
            if used_name in phrase_dict:
                print ("[Warning]: naming conflict of phrase:", phrase_name)
                continue
            
            pattern = utils.preproc(pattern).split(' ')
            pattern_chunk_list = self._extract_simple_rules(pattern)
            mapped_value = mapped_value.strip('()"')
            if mapped_value == "":
                mapped_value = 0
            elif mapped_value[0] == '$':
                mapped_value = int(mapped_value[1:])

            phrase_dict[used_name] = {'pattern':pattern_chunk_list, 'mapped_value':mapped_value}

            if used_name not in phrase_names_topology:
                phrase_names_topology[used_name] = {'in':set(), 'out':set()}
            for pattern_chunk in pattern_chunk_list:
                if pattern_chunk[0:2] == '${' and pattern_chunk[0:3] != '${.':
                    phrase_names_topology[used_name]['in'].add(pattern_chunk)
                    if pattern_chunk not in phrase_names_topology:
                        phrase_names_topology[pattern_chunk] = {'in':set(), 'out':set()}
                    phrase_names_topology[pattern_chunk]['out'].add(used_name)
        
        # 3. 根据短语规则之间的引用关系进行拓扑排序和分层；
        phrase_names_in_order = []
        no_incoming_nodes = set()
        for phrase_name in phrase_names_topology:
            if not phrase_names_topology[phrase_name]['in']: #len(phrase_names_topology[phrase_name]['in']) == 0:
                no_incoming_nodes.add(phrase_name)
        while no_incoming_nodes: #len(no_incoming_nodes) > 0:
            phrase_names_in_order.append(list(no_incoming_nodes))
            for n in phrase_names_in_order[-1]:
                no_incoming_nodes.remove(n)
                for m in phrase_names_topology[n]['out'].copy():
                    phrase_names_topology[m]['in'].remove(n)
                    phrase_names_topology[n]['out'].remove(m)
                    if not phrase_names_topology[m]['in']: #len(phrase_names_topology[m]['in']) == 0:
                        no_incoming_nodes.add(m)

        return phrase_dict, phrase_names_in_order
    
    def _analysis_lex(self, domain_lex):
        r"""
        从规则主文件读取所有的短语规则。
        
        规则文件：
            1. 支持"#include lex_file"的操作，类似c语言。
            2. ‘#’开头的其他行表示注释内容，所以目前仅支持行注释。

        Arguments:
            main_lex (path): 短语规则主文件路径；
        Returns:
            contents (list)：所有有效的规则文本；
        """
        contents = []
        pwd = '/'.join(domain_lex.split('/')[:-1])
        with open(domain_lex, 'r') as f:
            line_number = 0
            for line in f:
                line = line.strip('\n\r\t ;')
                line_number += 1

                if line == "": continue
                elif line[0] == '#':
                    if line.startswith('#include '):
                        include_filepath = line.split(' ')[1].strip('"')
                        tmp = include_filepath.split('/')
                        if tmp[0] == '${pwd}':
                            include_filepath = os.path.join(pwd, '/'.join(tmp[1:]))
                        contents += self._analysis_lex(include_filepath)
                    else:
                        contents.append(line)
                elif line[0] == '%':
                    pass
                else:
                    contents.append(line)
        return contents

    def _extract_simple_rules(self, pattern):
        r"""
        读取pattern的正则文本，自动分段。
        比如：
            pattern = ${.paper}(发表|刊登)?
            pattern_chunk_list = ['${.paper}', '(', '发', '表', '|', '刊', '登', ')?']

        Arguments:
            pattern (lex text): 正则文本
        Returns:
            pattern_chunk_list (list): 自动分段后的正则文本
        """
        pattern_chunk_list = []
        max_length = len(pattern)
        idx = 0
        while idx < max_length:
            cur_word = pattern[idx]
            if cur_word == ')':
                if idx+1 < max_length and pattern[idx+1] in ['?']: #['*', '+', '?']:
                    pattern_chunk_list.append(''.join(pattern[idx:idx+2]))
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
                    if end_idx+1 < max_length and pattern[end_idx+1] in ['?']: #['*', '+', '?']:
                        pattern_chunk_list.append(''.join(pattern[idx:end_idx+2]))
                        idx = end_idx+2
                    else:
                        pattern_chunk_list.append(''.join(pattern[idx:end_idx+1]))
                        idx = end_idx+1
            elif cur_word == ' ':
                idx += 1
            else:
                if idx+1 < max_length and pattern[idx+1] in ['?']: #['*', '+', '?']:
                    pattern_chunk_list.append(''.join(pattern[idx:idx+2]))
                    idx += 2
                else:
                    pattern_chunk_list.append(cur_word)
                    idx += 1

        return pattern_chunk_list

    def _build_vocab(self):
        r"""
        从数据中读取文本，构建FST的输入输出词表。
        
        Dependencies:
            phrase_dict (dict)：短语规则数据；
            fst_path (dir): FST存储路径
        
        Returns:
            isyms_table (fst sym): FST输入词表。
            osyms_table (fst sym): FST输出词表。
        """
        phrase_dict = self.phrase_dict
        isyms = ["<eps>"]
        label_voc = {}
        osyms = ["<eps>", "<unk>", "<name>"]
        word_voc = {} #["<unk>"] #<unk> should be defined manually
        for rule_name in phrase_dict:
            chunk_list = phrase_dict[rule_name]['pattern']
            for word in chunk_list:
                if word[0] not in ['(', ')', '|']:
                    word = word.strip('?') #word.strip('?+*')
                    if word.startswith('${.'):  # for DB word list
                        word = word.lower()
                    word_voc[word] = 1
            word_voc["<name>"+'_'+rule_name] = 1
        osyms = osyms + list(word_voc)

        # 短语规则的匹配取值标记：
        # 1. '$' 表示取对应位置的词，一般和argument_count(左括号计数)配合使用，比如 phrase1=(${int})(${单位}) => ("$1")；
        # 2. '_' 表示占位符，是为了让输入输出对齐，比如上面一行的例子中“(${单位})”需要占位；
        # 3. type(mapped_value) != int，即取值mapped_value是一个字符串，比如 phrase1=(我国) => ("中国");
        for rule_name in phrase_dict:
            pattern_chunk_list = phrase_dict[rule_name]['pattern']
            mapped_value = phrase_dict[rule_name]['mapped_value']
            label_voc[rule_name] = 1

            segment_stack = [{}]
            if mapped_value == 0:
                segment_stack[0]['value'] = '$'
            elif type(mapped_value) != int :
                segment_stack[0]['value'] = mapped_value
            else:
                segment_stack[0]['value'] = '_'
            argument_count = 0
            for word in pattern_chunk_list:
                if word == '(':
                    argument_count += 1
                    if argument_count == mapped_value:
                        segment_stack.append({'value':'$'})
                    else:
                        segment_stack.append({'value':segment_stack[-1]['value']})
                elif word[0] == ')':
                    segment_stack.pop()
                elif word == '|':
                    pass
                else:
                    label_voc[segment_stack[-1]['value']] = 1
        isyms = isyms + list(label_voc)

        isyms_table = fst.SymbolTable()
        for idx,val in enumerate(isyms):
            isyms_table[val] = idx
        isyms_table.write(os.path.join(self.fst_path,"isyms.fst"))
        osyms_table = fst.SymbolTable()
        for idx,val in enumerate(osyms):
            osyms_table[val] = idx
        osyms_table.write(os.path.join(self.fst_path,"osyms.fst"))

        return isyms_table, osyms_table

    def create_fsts(self):
        r"""
        从数据中读取短语正则，构建FST:
            0.  input:      self.phrase_dict
                output:     merged_fst
            1. 循环：对单个关于规则构建FST。
            2. 冲突检测：在FST网络压缩时，同一条规则路径对应的输出应该是唯一的，记录冲突规则并植入解歧义的特殊符号("<name>_xxx")，以备后续特殊处理；
            3. 按拓扑分层顺序，依次：
                3.1. merge 每一层中的FST成为一个整体的FST。
                3.2. 压缩整体的FST网络。
                3.3. 复原解歧义的特殊符号

        Dependencies:
            phrase_dict (dict)：短语规则数据；
            isyms_table (fst sym): FST输入词表。
            osyms_table (fst sym): FST输出词表。
            fst_path (dir): FST存储路径
        """
        phrase_dict = self.phrase_dict
        phrase_names_in_order = self.phrase_names_in_order
        isyms_table = self.isyms
        osyms_table = self.osyms
        weight = self.weight
        # 1. 循环：对单个关于规则构建FST。
        start_time = time.time()
        all_rule_paths = {}
        all_rule_naming_state = {}
        for rule_name in phrase_dict:
            pattern_chunk_list = phrase_dict[rule_name]['pattern']
            mapped_value = phrase_dict[rule_name]['mapped_value']
            start_idx = 1
            phrase_fst = fst.StdTransducer(isyms=isyms_table, osyms=osyms_table)
            # 默认从start_idx开始
            segment_stack = [{'begin':start_idx, 'end':None}]
            cursor_head, cursor_tail = start_idx, start_idx+1
            phrase_fst.add_arc(0, cursor_head, rule_name, "<eps>", 0.5)
            all_rule_naming_state[rule_name] = 0
            if mapped_value == 0:
                segment_stack[0]['value'] = '$'
                phrase_fst.add_arc(cursor_head, cursor_tail, '<eps>', '<eps>', 0)
            elif type(mapped_value) != int :
                segment_stack[0]['value'] = '_' #mapped_value
                phrase_fst.add_arc(cursor_head, cursor_tail, mapped_value, '<eps>', 0) # NOTE !!! a potential error, please refer to from_lex_to_fst_exportRule_norm.py
            else:
                segment_stack[0]['value'] = '_'
                phrase_fst.add_arc(cursor_head, cursor_tail, '<eps>', '<eps>', 0)
            cursor_head, cursor_tail = cursor_tail, cursor_tail+1
            segment_stack[0]['begin'] = cursor_head
            argument_count = 0
            for word in pattern_chunk_list:
                # 碰到'('，表示一个 segment 的开始；
                if word == '(':
                    argument_count += 1
                    if argument_count == mapped_value:
                        segment_stack.append({'begin':cursor_tail, 'end':None, 'value':'$'})
                    else:
                        segment_stack.append({'begin':cursor_tail, 'end':None, 'value':segment_stack[-1]['value']})
                    segment_stack[-1]['head_arc'] = [cursor_head, cursor_tail]
                    cursor_head = cursor_tail
                    cursor_tail += 1
                # 碰到')'，表示一个 segment的结束
                elif word[0] == ')':
                    if segment_stack[-1]['end'] == None:
                        segment_stack[-1]['end'] = cursor_head
                    else:
                        phrase_fst.add_arc(cursor_head, segment_stack[-1]['end'], '<eps>', '<eps>', 0)
                        cursor_head = segment_stack[-1]['end']
                    last_segment_value = segment_stack[-2]['value']
                    phrase_fst.add_arc(segment_stack[-1]['head_arc'][0], segment_stack[-1]['head_arc'][1], '<eps>', '<eps>', 0)
                    if word == ')?':
                        phrase_fst.add_arc(segment_stack[-1]['begin'], segment_stack[-1]['end'], '<eps>', '<eps>', 0)
                    segment_stack.pop()
                # 碰到'|'，表示在现有segment里面重新开一条并行的路径
                elif word == '|':
                    if segment_stack[-1]['end'] == None:
                        segment_stack[-1]['end'] = cursor_head
                    else:
                        phrase_fst.add_arc(cursor_head, segment_stack[-1]['end'], '<eps>', '<eps>', 0)
                    cursor_head = segment_stack[-1]['begin']
                # 碰到 word，表示加一条边；
                else:
                    if word[-1] == '?':
                        phrase_fst.add_arc(cursor_head, cursor_tail, '<eps>', '<eps>', 0)
                        word = word[:-1]
                    else:
                        pass
                    if word.startswith('${.'):  # for DB word list
                        word = word.lower()
                    next_state = self.add_arc(phrase_fst, cursor_head, cursor_tail, word, segment_stack[-1]['value'], weight)
                    cursor_head = cursor_tail
                    cursor_tail = next_state
            if segment_stack[-1]['end'] == None:
                segment_stack[-1]['end'] = cursor_head
            else:
                phrase_fst.add_arc(cursor_head, segment_stack[-1]['end'], '<eps>', '<eps>', 0)
            phrase_fst[segment_stack[-1]['end']].final = True
            self.FSTs[rule_name] = phrase_fst
            #self._draw_fst(phrase_fst, 'ph/'+rule_name.strip("${<>}"))
        
        # 2. 冲突检测：在FST网络压缩时，同一条规则路径对应的输出应该是唯一的，记录冲突规则并植入解歧义的特殊符号，以备后续特殊处理；
        for rule_name in self.FSTs:
            phrase_fst = self.FSTs[rule_name]
            for i, path in enumerate(phrase_fst.paths()):
                path_istring_list = [phrase_fst.isyms.find(arc.ilabel) for arc in path if arc.ilabel != 0]
                path_ostring_list = [phrase_fst.osyms.find(arc.olabel) for arc in path if arc.olabel != 0]
                path_istring = ' '.join(path_istring_list)
                path_ostring = ' '.join(path_ostring_list)
                if path_ostring not in all_rule_paths:
                    all_rule_paths[path_ostring] = {}
                if path_istring not in all_rule_paths[path_ostring]:
                    all_rule_paths[path_ostring][path_istring] = {}
                all_rule_paths[path_ostring][path_istring][rule_name] = 1
        conflict_rule_names = set()
        for in_snt in all_rule_paths:
            if len(all_rule_paths[in_snt]) > 1:
                for out_snt in all_rule_paths[in_snt]:
                    conflict_rule_names |= set(list(all_rule_paths[in_snt][out_snt]))
            else:
                for out_snt in all_rule_paths[in_snt]:
                    if len(all_rule_paths[in_snt][out_snt]) > 1:
                        conflict_rule_names |= set(list(all_rule_paths[in_snt][out_snt]))
        print ("#constraints with conflicts: %s" % (len(conflict_rule_names)))
        if len(conflict_rule_names)>0:
            print(conflict_rule_names)
        for rule_name in self.FSTs:
            if rule_name in conflict_rule_names:
                phrase_fst = self.FSTs[rule_name]
                for state in phrase_fst.states:
                    if state.stateid == all_rule_naming_state[rule_name]:
                        for arc in state.arcs:
                            arc.olabel = phrase_fst.osyms['<name>_'+rule_name]
                            break

        # 3. 按拓扑分层顺序，依次：
        #     3.1. merge 每一层中的FST成为一个整体的FST。
        #     3.2. 压缩整体的FST网络。
        #     3.3. 复原解歧义的特殊符号
        # merge FSTs and minimize it
        for rule_idx, rule_names in enumerate(phrase_names_in_order): # NOTE !!!
            merged_fst = fst.StdTransducer(isyms=isyms_table, osyms=osyms_table)
            final_state_idx = 1
            start_idx = 2
            for rule_name in rule_names: # NOTE !!!
                merged_fst.add_arc(0, start_idx, '<eps>', '<eps>', 0)
                t = self.FSTs[rule_name]
                current_state_id = 0
                for state in t.states:
                    current_state_id = max([current_state_id, state.stateid + start_idx])
                    if bool(state.final) == True:
                        end_state_id = state.stateid + start_idx
                    for arc in state.arcs:
                        merged_fst.add_arc(state.stateid + start_idx, arc.nextstate + start_idx, t.isyms.find(arc.ilabel), t.osyms.find(arc.olabel), arc.weight)
                merged_fst.add_arc(end_state_id, final_state_idx, '<eps>', '<eps>', 0)
                start_idx = current_state_id + 1
            merged_fst[final_state_idx].final = True

            merged_fst.remove_epsilon()
            merged_fst = merged_fst.inverse()
            merged_fst = merged_fst.determinize()
            merged_fst.minimize()
            merged_fst = merged_fst.inverse()

            for state in merged_fst.states:
                for arc in state.arcs:
                    if merged_fst.osyms.find(arc.olabel).startswith("<name>_"):
                        arc.olabel = 0
            
            merged_fst.write(self.fst_path+'/'+str(rule_idx)+'_'+str(len(rule_names))+'.fst')

            #self._draw_fst(merged_fst, 'ph/'+str(rule_idx)+'_'+str(len(rule_names)))
    
    def add_arc(self, my_fst, fromstate, tostate, word, label, weight):
        '''
        这是一个预留，以后如果碰到很复杂的 word，比如支持了 word{n,m} 这一类的正则写法，可以扩展这个函数。
        '''
        next_state = tostate+1
        my_fst.add_arc(fromstate, tostate, label, word, weight)
        return next_state
        
    def _draw_fst(self, fst_model, name):
        out = fst_model.draw(isyms=self.isyms, osyms=self.osyms)
        dot_file = open('dot/'+name+'.dot', 'wb')
        dot_file.write(out)
        dot_file.close()
        os.system('dot -Tpdf -O '+'dot/'+name+'.dot')

class LexiconFSTBuilder():
    r"""
    从词库目录读取所有的词库文件，先对每一个词库分别单独构建FST，然后合并为一个整体的词匹配FST。

    Arguments:
        database_path (dir): 词库目录名，目录中有很多词库文件，比如文件“city_name”中每一行是一个城市名。
        fst_path (dir): FST存储路径。
        weight (float): FST边权重；
        thread_num (int): 对所有词库分别构建FST时使用的线程数量；

        FSTs (dict): 该字典包含每一个词库的FST；
        lexicon_tables (dict): 词库数据，比如 {"${.city_name}":[("上", "海"), ("苏", "州"), ("new", "york"), ...], ...};
        conflict_words (set): 多义词，即在多个词库中同时出现的词，构建FST时需要对这些词特殊对待。
    """
    def __init__(self, database_path, fst_path, weight=-1, thread_num=6):
        self.lexicon_tables, self.conflict_words = self._read_data(database_path)
        self.FSTs = {}
        self.weight = weight
        self.thread_num = thread_num
        
        #清空目标路径
        if not os.path.isdir(fst_path):
            os.makedirs(fst_path)
        else:
            for filename in os.listdir(fst_path):
                targetFile = os.path.join(fst_path,filename)
                if os.path.isfile(targetFile):
                    os.remove(targetFile)
        self.fst_path = fst_path
        
        self.isyms, self.osyms = self._build_vocab()
    
    def _read_data(self, database_path):
        r"""
        从词库目录读取所有的词库文件。

        Arguments:
            database_path (dir): 词库目录名，目录中有很多词库文件，比如文件“city_name”中每一行是一个城市名。
        Returns:
            lexicon_tables (dict): 词库数据，比如 {"${.city_name}":[("上", "海"), ("苏", "州"), ("new", "_", "york"), ...], ...};
            conflict_words (set): 多义词，即在多个词库中同时出现的词，构建FST时需要对这些词特殊对待。
        """
        lexicon_tables = {}
        reverse_lexicon_tables = {}
        lists = os.listdir(database_path)
        for lexicon_name in lists:
            filepath = os.path.join(database_path, lexicon_name)
            if os.path.isdir(filepath): #如果filepath是目录
                pass
            elif os.path.isfile(filepath):
                if lexicon_name.endswith('.txt'):
                    lexicon_name = lexicon_name[:- len('.txt')]
                data_file = open(filepath, 'r')
                lexicon_tables[lexicon_name] = {}
                for data_line in data_file:
                    data_line = data_line.strip()
                    data_line = utils.preproc(data_line)
                    data_line = data_line.split(' ')
                    lexicon_tables[lexicon_name][tuple(data_line)] = 1
                    if tuple(data_line) not in reverse_lexicon_tables:
                        reverse_lexicon_tables[tuple(data_line)] = {lexicon_name:1}
                    else:
                        reverse_lexicon_tables[tuple(data_line)][lexicon_name] = 1
                data_file.close()
        conflict_words = set()
        for words in reverse_lexicon_tables:
            if len(reverse_lexicon_tables[words]) > 1:
                conflict_words.add(words)
        for lexicon_name in lexicon_tables:
            lexicon_tables[lexicon_name] = list(lexicon_tables[lexicon_name])
        return lexicon_tables, conflict_words

    def _build_vocab(self):
        r"""
        从数据中读取文本，构建FST的输入输出词表。
        
        Dependencies:
            lexicon_tables (dict): 词库数据，比如 {"${.city_name}":[("上", "海"), ("苏", "州"), ("new", "_", "york"), ...], ...};
            fst_path (dir): FST存储路径
        
        Returns:
            isyms_table (fst sym): FST输入词表。
            osyms_table (fst sym): FST输出词表。
        """
        lexicon_tables = self.lexicon_tables
        isyms = ["<eps>"]
        word_voc = {} #["<unk>"] #<unk> should be defined manually
        osyms = ["<eps>", "<unk>"]

        for name in lexicon_tables:
            word_voc['<name>_'+name] = 1
            for words_list in lexicon_tables[name]:
                label = '${.'+name+'}'
                for word in words_list:
                    word_voc[word] = 1
                if label not in isyms:
                    isyms += [label]

        osyms = osyms + list(word_voc)
        isyms_table = fst.SymbolTable()
        for idx,val in enumerate(isyms):
            isyms_table[val] = idx
        osyms_table = fst.SymbolTable()
        for idx,val in enumerate(osyms):
            osyms_table[val] = idx
        isyms_table.write(os.path.join(self.fst_path,"isyms.fst"))
        osyms_table.write(os.path.join(self.fst_path,"osyms.fst"))
        return isyms_table, osyms_table

    def create_fsts(self):
        r"""
        从数据中读取文本，构建FST:
            1. 循环：对单个词库构建FST；并针对多义词植入解歧义的特殊符号（“<name>_xxx”），保证FST网络可以压缩。
            2. merge 所有的单FST成为一个整体的FST。
            3. 压缩整体的FST网络。
            4. 复原解歧义的特殊符号。

        Dependencies:
            lexicon_tables (dict): 词库数据，比如 {"${.city_name}":[("上", "海"), ("苏", "州"), ("new", "_", "york"), ...], ...};
            conflict_words (set): 多义词，即在多个词库中同时出现的词，构建FST时需要对这些词特殊对待。
            isyms_table (fst sym): FST输入词表。
            osyms_table (fst sym): FST输出词表。
            fst_path (dir): FST存储路径
        """
        # 1. 循环：对单个词库构建FST；并针对多义词植入解歧义的特殊符号（“<name>_xxx”），保证FST网络可以压缩。
        names = list(self.lexicon_tables)
        cursor = 0
        threads = []
        while cursor < len(names):
            for i in range(self.thread_num):
                if cursor + i >= len(names):
                    break
                t = threading.Thread(target=self._create_fst, args=(names[cursor+i],)) 
                threads.append(t)
            cursor += self.thread_num

            for i in range(len(threads)):
                threads[i].start()

            for i in range(len(threads)):
                threads[i].join()

            threads = []
        
        # 2. merge 所有的单FST成为一个整体的FST。
        merged_fst = fst.StdTransducer(isyms=self.isyms, osyms=self.osyms)
        final_state_idx = 1
        start_idx = 2
        for rule_name in self.FSTs:
            merged_fst.add_arc(0, start_idx, '<eps>', '<eps>', 0)
            t = self.FSTs[rule_name]
            current_state_id = 0
            for state in t.states:
                current_state_id = max([current_state_id, state.stateid + start_idx])
                if bool(state.final) == True:
                    end_state_id = state.stateid + start_idx
                for arc in state.arcs:
                    merged_fst.add_arc(state.stateid + start_idx, arc.nextstate + start_idx, t.isyms.find(arc.ilabel), t.osyms.find(arc.olabel), arc.weight)
            merged_fst.add_arc(end_state_id, final_state_idx, '<eps>', '<eps>', 0)
            start_idx = current_state_id + 1
        merged_fst[final_state_idx].final = True
        
        # 3. 压缩整体的FST网络。
        merged_fst.remove_epsilon()
        merged_fst = merged_fst.inverse()
        merged_fst = merged_fst.determinize()
        merged_fst.minimize()
        merged_fst = merged_fst.inverse()

        for state in merged_fst.states:
            for arc in state.arcs:
                if merged_fst.osyms.find(arc.olabel).startswith("<name>_"):
                    arc.olabel = 0
        
        merged_fst.write(self.fst_path+'/db_all.fst')
        
        #self._draw_fst(merged_fst, 'wb/db_all')

    def _create_fst(self, name):
        r"""
        对单个词库构建FST；并针对多义词植入解歧义的特殊符号（“<name>_xxx”），保证FST网络可以压缩。

        Dependencies:
            lexicon_tables (dict): 词库数据，比如 {"${.city_name}":[("上", "海"), ("苏", "州"), ("new", "_", "york"), ...], ...};
            conflict_words (set): 多义词，即在多个词库中同时出现的词，构建FST时需要对这些词特殊对待。
            isyms_table (fst sym): FST输入词表。
            osyms_table (fst sym): FST输出词表。
        Arguments:
            name (string): 词库名称，比如“city_name”。
        """
        lexicon_tables = self.lexicon_tables
        conflict_words = self.conflict_words
        isyms_table = self.isyms
        osyms_table = self.osyms
        weight = self.weight
        replace = fst.StdTransducer(isyms=isyms_table,osyms=osyms_table)
        fromstate = 3
        label = '${.'+name+'}'
        naming_flag = False
        for strs in lexicon_tables[name]:
            if strs not in conflict_words:
                if not naming_flag:
                    naming_flag = True
                    replace.add_arc(0, 1, label, '<eps>', 0.5)
                start_idx = 1
            else:
                replace.add_arc(0, fromstate, label, '<name>_'+name, 0.5)
                start_idx = fromstate
                fromstate += 1
            word_count = len(strs)
            if word_count == 1:
                replace.add_arc(start_idx, 2, '<eps>', strs[0], weight)
            else:
                for idx,word in enumerate(strs):
                    if idx == 0:
                        replace.add_arc(start_idx, fromstate, '<eps>', word, weight)
                    elif idx == word_count - 1:
                        replace.add_arc(fromstate, 2, '<eps>', word, weight)
                        fromstate += 1
                    else:
                        replace.add_arc(fromstate, fromstate+1, '<eps>', word, weight)
                        fromstate += 1
        replace[2].final = True

        self.FSTs[name] = replace

        #self._draw_fst(replace, 'wb/'+name)

    def _draw_fst(self, fst_model, name):
        out = fst_model.draw(isyms=self.isyms, osyms=self.osyms)
        dot_file = open('dot/'+name+'.dot', 'wb')
        dot_file.write(out)
        dot_file.close()
        os.system('dot -Tpdf -O '+'dot/'+name+'.dot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d','--domain',dest='domain',action='store',required=True,help='which domain')
    parser.add_argument('-w','--weight',dest='weight',action='store',metavar='number',required=False,default=-1,type=float,help='weight number')
    args = parser.parse_args()
    wei = args.weight

    # step1: build fst for lexicons
    database_path=PATH_TO_TAGGER[args.domain]
    fst_path=os.path.join(PATH_TO_FST[args.domain],'lexicon')
    lexicon_fst = LexiconFSTBuilder(database_path, fst_path, wei)
    lexicon_fst.create_fsts()

    # step2: build fst for constraints
    main_lex=PATH_TO_CONSTRAINT[args.domain]
    fst_path=os.path.join(PATH_TO_FST[args.domain],'constraint')
    constraint_fst=PhraseFSTBuilder(main_lex=main_lex,fst_path=fst_path,weight=wei)
    constraint_fst.create_fsts()
    
