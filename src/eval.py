#!/usr/bin/env python
#coding=utf8
import os,sys,json,time,argparse
from utils import *
from global_vars import *
from models.seq_rnn_model import SeqRNNModel
from models.seq_rnn_crf_model import SeqRNNCRFModel
from models.focus_model import FocusModel
from models.decoder_and_eval import *
from models.loss_function import *
import torch
from torch.autograd import Variable
import gpu_selection
from collections import defaultdict

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--domain',choices=['SpeechLab','AISpeech'],default='SpeechLab')
    parser.add_argument('-m','--model',default='RNN',help='choose model from rnn,rnn+crf,focus models')
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--beos',action='store_true')
    parser.add_argument('-idpt','--input_dropout',type=float,default=0.2)
    parser.add_argument('-odpt','--dropout',type=float,default=0.5)
    parser.add_argument('--use_word_embeddings',action='store_true')
    parser.add_argument('--bidirection',action='store_true')
    parser.add_argument('-l','--layers',type=int,default=1)
    parser.add_argument('-c','--cell',choices=['LSTM','GRU'],default='LSTM')
    parser.add_argument('-e','--embedding_size',type=int,default=256,help='use an additional embedding layer before network input')
    parser.add_argument('-hs','--hidden_size',type=int,default=512)
    parser.add_argument('-bs','--batch_size',type=int,default=64,help='size of mini-batches')
    parser.add_argument('-voc','--voc_ratio',type=float,default=0.939)
    parser.add_argument('--post_attention',action='store_true')
    parser.add_argument('--annotation',action='store_true',help='eval data whether has annotation')
    args=parser.parse_args()
    args.model=args.model.lower()
    if args.model not in ['rnn','rnn+crf','seq2seq','attention','focus']:
        raise ValueError('[Error]: unknown model name',args.model)
    if args.model in ['seq2seq','attention','focus']:
        args.beos=True
        args.seq2seq=True
    else:
        args.seq2seq=False
    if args.cuda:
        deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu()
        print("Valid GPU list: %s ; GPU %d (%s) is auto selected." % (valid_gpus, deviceId, gpu_name))
        torch.cuda.set_device(deviceId)
    args.cuda=args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('We will use cuda ... ...')
        torch.cuda.manual_seed(1)
    torch.manual_seed(1)

    start_time=time.time()
    # load vocabulary and tagset
    print('Load vocabulary dict and slots,tags dict:...')
    voc_dict,reverse_voc_dict=load_vocabulary(PATH_TO_VOC[args.domain],args.voc_ratio,args.beos)
    slot_tag_dict,reverse_slot_tag_dict=load_tagset(PATH_TO_SLOT_TAGSET[args.domain],seq2seq=args.seq2seq)
    sentence_tag_dict,reverse_sentence_tag_dict=load_tagset(PATH_TO_SENTENCE_TAGSET[args.domain])
    # prepare eval data
    evalset=load_evaldata(PATH_TO_EVAL_DATA[args.domain],voc_dict,args.beos,PATH_TO_STOP_PATTERN[args.domain],annotation=args.annotation)
    dataloader=Dataset(evalset,1,0) # 对测试数据进行封装
    print('Size of eval data is',len(evalset))
    print('Size of vocabulary is',len(voc_dict))
    pad_num=len(voc_dict)
    ignore_slot_index=-100 if args.model in ['rnn','rnn+crf'] else len(slot_tag_dict)
    print('Prepare resource time:',(time.time()-start_time))
    if args.model=='rnn':
        eval_net=SeqRNNModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
    elif args.model=='rnn+crf':
        eval_net=SeqRNNCRFModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
    elif args.model in ['seq2seq','attention','focus']:
        eval_net=FocusModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
    eval_net.load_module(os.path.join(PATH_TO_MODELS[args.domain],'train.'+args.model+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))
    if args.cuda:
        eval_net.cuda()
    
    eval_net.eval()
    file_path=open(os.path.join(PATH_TO_MODELS[args.domain],'eval.result'),'w')
    if args.annotation:
        error_path=open(os.path.join(PATH_TO_MODELS[args.domain],'eval.error'),'w')
        slot_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0})
        sentence_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0})
        whole_dict={'Num':0,'TP':0,'SubSet':0,'SuperSet':0}
        char_path=os.path.join(PATH_TO_MODELS[args.domain],'eval.char.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.score')
        slot_path=os.path.join(PATH_TO_MODELS[args.domain],'eval.slot.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.score')
    domain_ontology=Ontology(PATH_TO_ONTOLOGY[args.domain])
    post_processor=PostProcessor(PATH_TO_CONSTRAINT_MAPPINGS[args.domain],PATH_TO_POST_PROCESS_DIR[args.domain],domain_ontology)
    for step,(data_sample,eval_length_list) in enumerate(dataloader.get_mini_batches(batch_size=args.batch_size,data=dataloader.data,pad_num=pad_num,ignore_slot_index=ignore_slot_index,shuffle=False)):
        if args.annotation:
            eval_raw_data,padded_eval_data,ref_semantics=data_sample
        else:
            eval_raw_data,padded_eval_data=data_sample
        if args.cuda:
            padded_eval_data=padded_eval_data.cuda()
        if args.model=='rnn':
            eval_slot_scores,eval_sentence_scores=eval_net(padded_eval_data,eval_length_list)
            # kbest decoder
            # eval_slot_result,eval_sentence_result=kbest_decoder(eval_slot_scores,eval_sentence_scores,eval_length_list,kbest=5,ignore_slot_index=ignore_slot_index)
            # bio decoder
            eval_slot_result,eval_sentence_result=bio_decoder(eval_slot_scores,eval_sentence_scores,eval_length_list,reverse_slot_tag_dict,kbest=1,ignore_slot_index=ignore_slot_index,penalty=-1000000)
            # eval_slot_result : [batch_size,kbest,max_sequence_length]
            # eval_sentence_result : [batch_size,kbest]
            flag=True
        elif args.model=='rnn+crf':
            eval_slot_result,eval_sentence_result=eval_net(padded_eval_data,eval_length_list,ignore_slot_index=ignore_slot_index)
            eval_slot_result,eval_sentence_result=eval_slot_result.data.cpu(),eval_sentence_result.data.cpu()
            flag=False
        elif args.model in ['seq2seq','focus','attention']:
            eval_slot_result,_,_,eval_sentence_scores=eval_net.decoder_greed(padded_eval_data,eval_length_list,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
            _,eval_sentence_result=eval_sentence_scores.max(dim=1)
            flag=False
            eval_slot_result,eval_sentence_result=eval_slot_result.data.cpu(),eval_sentence_result.data.cpu()
        
        semantics_list=obtain_semantics(
            eval_slot_result,eval_sentence_result,
            eval_raw_data,reverse_slot_tag_dict,reverse_sentence_tag_dict,
            domain_ontology,kbest=flag,use_ontology=False,ignore_slot_index=ignore_slot_index,
            fp=file_path,post_processor=post_processor,encapsulate=args.annotation)
        if args.annotation:
            for i,each_ref in enumerate(ref_semantics):
                #对参考标注进行归一化
                current_semantic=post_processor.post_process_constraint(each_ref,semantic_type='recursive')
                current_semantic=post_processor.post_process_entity(current_semantic,semantic_type='recursive')
                ref_semantics[i]=current_semantic
            slot_dict,sentence_dict,whole_dict=compare_semantic(semantics_list,ref_semantics,slot_dict,sentence_dict,whole_dict,error_fp=error_path)
        else:
            for i,each_semantic_result in enumerate(semantics_list):
                print("Query: "+' '.join(eval_raw_data[i])+' =>\nParse: '+json.dumps(each_semantic_result,ensure_ascii=False)+'\n')
    if args.annotation:
        print_char_level_scores(slot_result_dict=None,sentence_result_dict=sentence_dict,path=char_path)
        print_slot_level_scores(slot_result_dict=slot_dict,sentence_result_dict=whole_dict,path=slot_path)
    file_path.close()
