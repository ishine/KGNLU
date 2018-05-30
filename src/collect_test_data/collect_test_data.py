#!/usr/bin/env python3
#coding=utf8
import sys,os,argparse,re,json
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from utils import *
from global_vars import *
import csv
from collections import defaultdict

def parse_csv(filename):
    # 问卷收集的数据部分出了点错，出现[题目填空]的异常选项，导致[选项填空]死翘翘了
    csv_file=csv.reader(open(filename,'r'))
    title_list,result_dict=[],{}
    for i,each in enumerate(csv_file):
        each=each[4:]
        if i==0:
            for col in each:
                if col.strip()=='': # prevent blank title
                    continue
                title_list.append(col)
        else:
            structure=zip(title_list,each)
            splits=split_question(structure)
            result_dict=deal_with_splits(splits,result_dict)
    return result_dict

def split_question(structure):
    tmp_list,final_list,count=[],[],1
    for each in structure:
        if each[0].startswith(str(count)):
            tmp_list.append(each)
        else:
            final_list.append(tmp_list)
            count+=1
            tmp_list=[]
            if each[0].startswith(str(count)):
                tmp_list.append(each)
            else:
                count=int(each[0][:each[0].find('.')])
                tmp_list.append(each)
                print('[Warning]:non-contiguous data!')
    if tmp_list!=[]:
        final_list.append(tmp_list)
    return final_list

def deal_with_splits(splits,result=None):
    if not result:
        result={}
    for i,each in enumerate(splits):
        directive=each[0][0]
        assert (i+1)==int(directive[:directive.find('.')])
        no_info=False
        if '[题目填空]' in directive:
            missing_value=each[0][1].strip().split(',')
            missing_value_idx=0
            slots=directive[:directive.find('[题目填空]')].split('=>')[1].strip()
            slots=slots.split(' ， ') if ' ， ' in slots else slots.split(' , ')
            original_slots,replaced_slots={},{}
            sentence,current_slot='',''
            for each_slot in slots:
                original_slots[each_slot.split('=')[0].strip()]=each_slot.split('=')[1].strip()
            to_be_filled=False
            for pair in each[1:]:
                col=pair[0]
                value=pair[1]
                if '改写为' in col and '[选项填空]' not in col: 
                    to_be_filled=True # 准备接受值
                    continue
                elif '改写为' in col and '[选项填空]' in col:
                    assert to_be_filled==True
                    to_be_filled=False
                    if value.strip()=='':
                        no_info=True
                        break
                    else:
                        no_info=False
                        sentence+=value.strip()
                elif ':slot：' in col and '[选项填空]' not in col:
                    if to_be_filled:#还有sentence或slot等待被赋值
                        if sentence=='':#sentence还没有被赋值,查询missingvalue
                            if missing_value[missing_value_idx]=='':
                                to_be_filled=False
                                no_info=True
                                break
                            else:
                                no_info=False
                                sentence+=missing_value[missing_value_idx]
                                missing_value_idx+=1
                        else:#上个slot还没有被赋值
                            replaced_slots[current_slot]=original_slots[current_slot] if (len(missing_value)<=missing_value_idx or missing_value[missing_value_idx]=='') else missing_value[missing_value_idx]
                            missing_value_idx+=1
                    to_be_filled=True
                    current_slot=col[col.find(':slot：')+len(':slot：'):].split('=')[0].strip()
                    replaced_slots[current_slot]=''
                    if current_slot not in original_slots:
                        raise ValueError('[Error]:while dealing with slot name',current_slot,'during question',str(i+1))
                elif ':slot：' in col and '[选项填空]' in col:
                    to_be_filled=False
                    slot_name=col[col.find(':slot：')+len(':slot：'):].split('=')[0].strip()
                    if slot_name==current_slot and slot_name in original_slots:
                        replaced_slots[slot_name]=original_slots[slot_name] if value.strip()=='' else value.strip()
                    else:
                        raise ValueError('[Error]:while dealing with slot name',slot_name,'during question',str(i+1))
                else:
                    raise ValueError('[Error]:unknown pattern in title',col,'during question',str(i+1))
            if to_be_filled:
                replaced_slots[current_slot]=original_slots[current_slot] if (len(missing_value)<=missing_value_idx or missing_value[missing_value_idx]=='') else missing_value[missing_value_idx]
        else:
            slots=directive[:directive.find(':改写为')].split('=>')[1].strip()
            slots=slots.split(' ， ') if ' ， ' in slots else slots.split(' , ')
            original_slots,replaced_slots={},{}
            for each_slot in slots:
                original_slots[each_slot.split('=')[0].strip()]=each_slot.split('=')[1].strip()
            sentence,current_slot='',''
            for pair in each:
                col=pair[0]
                value=pair[1]
                if '改写为' in col and '[选项填空]' not in col:
                    continue
                elif '改写为' in col and '[选项填空]' in col:
                    if value.strip()=='':
                        no_info=True
                        break
                    else:
                        no_info=False
                        sentence+=value.strip()
                elif ':slot：' in col and '[选项填空]' not in col:
                    current_slot=col[col.find(':slot：')+len(':slot：'):].split('=')[0].strip()
                    replaced_slots[current_slot]=''
                    if current_slot not in original_slots:
                        raise ValueError('[Error]:while dealing with slot name',current_slot,'during question',str(i+1))
                elif ':slot：' in col and '[选项填空]' in col:
                    slot_name=col[col.find(':slot：')+len(':slot：'):].split('=')[0].strip()
                    if current_slot==slot_name and slot_name in original_slots:
                        replaced_slots[slot_name]=original_slots[slot_name] if value.strip()=='' else value.strip()
                    else:
                        raise ValueError('[Error]:while dealing with slot name',slot_name,'during question',str(i+1))
                else:
                    raise ValueError('[Error]:unknown pattern in title',col,'during question',str(i+1))
        if not no_info:
            if str(i+1) not in result:
                result[str(i+1)]=[]
            result[str(i+1)].append(sentence+' => '+json.dumps(replaced_slots,ensure_ascii=False))
    return result

def combine_annotation(data,ref_annotation,outfile,domain='SpeechLab'):
    testdata=[]
    process=PreProcessor(PATH_TO_STOP_PATTERN[domain])
    with open(ref_annotation,'r') as infile,open(outfile,'w') as of:
        count=0
        for idx,line in enumerate(infile):
            line=line.strip()
            if line=='':
                continue
            count+=1
            if str(count) in data:
                annotation=line.split('=>')
                return_type,original_slots=annotation[2],json.loads(annotation[3].strip())
                for each_data in data[str(count)]:
                    sentence,slots=each_data.split(' => ')
                    sentence=process.preprocess(sentence.strip())
                    slots=json.loads(slots)
                    for each_slot in slots:
                        slots[each_slot]=process.rm_punc(slots[each_slot].lower())
                    sentence,flag=add_slots_to_sentence(sentence,slots)
                    if flag:
                        original_slots=replace_slot(original_slots,slots)
                        testdata.append(' => '.join([sentence,return_type,json.dumps(original_slots)]))
                        of.write(' => '.join([sentence,return_type,json.dumps(original_slots,ensure_ascii=False)])+'\n')
                    else:
                        print('[Warning]:slot value not used in',sentence,' => slots:',slots)
    return testdata

def add_slots_to_sentence(sentence,slots):
    for each in slots:
        value=slots[each]
        if value not in sentence:
            return sentence,False
        if each.startswith('${#'):
            sentence=sentence.replace(value,'('+value+':mainEntity='+each+')')
        else:
            sentence=sentence.replace(value,'['+value+':slot='+each+']')
    return sentence,True

def replace_slot(original_slots,slots):
    for each in slots:
        if each.startswith('${#'):
            original_slots['mainEntity']=slots[each]
        for i,each_dict in enumerate(original_slots['slots']):
            if each_dict['name']==each:
                each_dict['value']=slots[each]
                original_slots['slots'][i]=each_dict
                break        
        else:
            raise ValueError('[Error]:unrecognized slot name',each,'in',slots,'compared with',original_slots)    
    return original_slots

def build_testset(ifilename,ofilename,shuffle=False):
    with open(ifilename,'r') as infile,open(ofilename,'w') as outfile:
        testset=[]
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
                        tmp_bio_list.append('O')
                    char_list.extend(tmp_char_list)
                    bio_list.extend(tmp_bio_list)
            item.append((char_list,bio_list))
            testset.append(item)

        if shuffle:
            random.shuffle(testset)

        for each in testset:
            outfile.write('\n') # 用来区分每一个sample
            outfile.write(each[0]+'\n') # 第一个位置是sentence classification结果
            outfile.write(each[1]+'\n') # 第二个位置是解析的语义槽
            char_list,bio_list=each[2]
            assert len(char_list)==len(bio_list)
            for idx,_ in enumerate(bio_list):
                outfile.write(char_list[idx]+'\t'+bio_list[idx]+'\n')
        return len(testset)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--domain',default='SpeechLab')
    parser.add_argument('-f','--filename',default='KGNLU.csv') #垃圾问卷系统，需要把'#39;'替换为'\''
    parser.add_argument('--annotation',action='store_true')
    parser.add_argument('--testdata',action='store_true')
    args=parser.parse_args()

    file_name=os.path.join(os.path.dirname(PATH_TO_TEST_DATA[args.domain]),args.filename)
    ref_annotation=PATH_TO_TEST_DATA[args.domain]+'.sample'
    annotation_file=os.path.join(os.path.dirname(PATH_TO_TEST_DATA[args.domain]),'test.annotation.txt')
    if args.annotation:
        result=parse_csv(file_name) # result:{1:[...,...],2:[...,...],... ...} 数字对应题目号
        # print('Get result from csv file finished!')
        # count=0
        # for each in result:
        #     print(result[each])
        #     count+=len(result[each])
        # print(count)
        testdata=combine_annotation(result,ref_annotation,annotation_file,domain=args.domain)
        # print('Transform result to rules.annotation.txt format finished!')
    if args.testdata:
        build_testset(annotation_file,PATH_TO_TEST_DATA[args.domain],shuffle=False)
        # print('Transform annotation to testdata finished!')