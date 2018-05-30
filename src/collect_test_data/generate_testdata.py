#!/usr/bin/env python3
#coding=utf8
import sys,os,argparse,re
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from utils import *
from global_vars import *

def reformat_testdata(input_file,out):
    # 大 概 介 绍 一 下 (实验室:mainEntity=${#institute}) 建 立 过 程 => description => {"return_type": "description", "mainEntity": "实验室", "slots": [{"op": "eq", "name": "${#institute}", "value": "实验室"}]}
    # 变为
    # 大概介绍一下实验室建立过程 => ${#institute}=实验室 =>  description => {"return_type": "description", "mainEntity": "实验室", "slots": [{"op": "eq", "name": "${#institute}", "value": "实验室"}]}
    extract_main_entity=re.compile(r'\((.*?)\)')
    extract_slot=re.compile(r'\[(.*?)\]')
    with open(input_file,'r') as inf,open(out,'w') as of:
        for line in inf:
            line=line.strip()
            if line=='':
                continue
            split_line=line.split('=>')
            restriction=''
            sent,semantics=split_line[0],'=>'.join(split_line[1:])
            mainEntity=extract_main_entity.findall(sent)
            slot=extract_slot.findall(sent)
            for each in mainEntity:
                restriction=restriction+' , '+each.split('=')[1]+'='+each.split(':')[0]
                sent=sent.replace('('+each+')',each.split(':')[0])
            for each in slot:
                restriction=restriction+' , '+each.split('=')[1]+'='+each.split(':')[0]
                sent=sent.replace('['+each+']',each.split(':')[0])
            of.write(reverse_preproc(preproc(sent).split(' '))+' => '+restriction.lstrip(' ,')+' => '+semantics+'\n\n')

def design_questionnaire(input_file,outfile):
    with open(input_file,'r') as inf,open(outfile,'w') as of:
        for each in inf:
            each=each.strip()
            if each=='':
                continue
            each=each.split('=>')
            slots=each[1]
            slot_names=[slot_name.split('=')[0].strip() for slot_name in slots.split(',')]
            of.write('引导语：'+' => '.join(each[:2])+' [多选题]\n改写为：'+'_'*25+'\n')
            for each_slot in slot_names:
                of.write('slot：'+each_slot+'='+'_'*25+'\n')
            of.write('\n')


if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--domain',default='SpeechLab')
    parser.add_argument('--reformat',action='store_true')
    parser.add_argument('--questionnaire',action='store_true')
    args=parser.parse_args()

    input_file=os.path.join(PATH_TO_DATA[args.domain],'rules.test.annotation.txt')
    out=PATH_TO_TEST_DATA[args.domain]+'.sample'
    questionnaire=PATH_TO_TEST_DATA[args.domain]+'.questionnaire'

    if args.reformat:
        reformat_testdata(input_file,out)
    if args.questionnaire:
        design_questionnaire(out,questionnaire)
        
# 本问卷用来采集针对"智能语音实验室"领域提问的测试数据及标注，请根据引导语的询问意图，改写上下文的问法并填写你使用的语义槽(slot)值(如果完全没有改变引导语中的slot值可以不填)
# 注意事项：
# 1.引导语中由 => 引出的是slot信息，问句中非slot部分是上下文
# 2.slot填空框中填写的内容必须和改写语句中使用的值完全一致(不区分大小写)，比如你在问句中使用的是"俞老板"，slot填写时请填写"俞老板"而非"俞凯"
# 3.如果你在改写句子中完全没有修改引导语中的slot值，可以不填
# 4.改写时任何一个slot的信息不能丢失，比如问"实验室的本科生"，错误改写为"你们实验室有哪些人"缺少身份(identity=本科生)信息
# 5.所有提问针对的是我们智能语音实验室，引导语由语法规则生成，不够口语化，请先理解它所要询问的意图(引导语最后的"多选题"字样请忽略)
# 样例1：
# 引导语：俞凯去年发表了什么论文 => ${#person}=俞凯 , publication_date=去年
# 改写为：2016年钱教授发了哪些论文
# slot: ${#person}=钱教授
# slot: publication_date=2016年
# 样例2：
# 引导语: 实验室本科生有哪些 => ${#institute}=实验室 , identity=本科生
# 改写为: 智能语音实验室有哪些博士
# slot: ${#institute}=智能语音实验室
# slot: identity=博士
# 样例3：
# 引导语：Speaker Verification with Deep Features什么时候发表的 => ${paper}=Speaker Verification with Deep Features
# 改写为：Speaker Verification with Deep Features的发表时间是哪年
# slot: ${#paper}=_________________________(没有改动引导语的slot值可以不填)
