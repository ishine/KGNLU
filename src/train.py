#!/usr/bin/env python
#coding=utf8
import os,sys,argparse,json,time,random
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from global_vars import *
from utils import *
from models.seq_rnn_model import SeqRNNModel
from models.seq_rnn_crf_model import SeqRNNCRFModel
from models.focus_model import FocusModel
from models.decoder_and_eval import *
from models.loss_function import *
import gpu_selection

def main(argv):
    parser=argparse.ArgumentParser(description='This is a program to train the neural network for KG NLU!')
    parser.add_argument('-d','--domain',default='SpeechLab',choices=['AISpeech','SpeechLab'])
    parser.add_argument('--train',action='store_true',help='train model use entire training set')
    parser.add_argument('--validation',action='store_true',help='use validation dataset to choose hyperparameters or eval the model with dev dataset')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('-idpt','--input_dropout',type=float,default=0.2)
    parser.add_argument('-odpt','--dropout',type=float,default=0.5)
    parser.add_argument('--bidirection',action='store_true')
    parser.add_argument('-l','--layers',type=int,default=1)
    parser.add_argument('-c','--cell',choices=['LSTM','GRU'],default='LSTM')
    parser.add_argument('-e','--embedding_size',type=int,default=256,help='use an additional embedding layer before network input')
    parser.add_argument('--use_word_embeddings',action='store_true',help='whether use pre-trained word embeddings')
    parser.add_argument('-hs','--hidden_size',type=int,default=512)
    parser.add_argument('--optimizer',choices=['sgd','adam'],default='sgd')
    parser.add_argument('-lr','--learning_rate',type=float,default=0.1)
    parser.add_argument('-mt','--momentum',type=float,default=0)
    parser.add_argument('-wd','--weight_decay',type=float,default=0)
    parser.add_argument('-m','--model',default='RNN',help='choose model from RNN,RNN+CRF,Seq2Seq,Attention models\
        (Seq2Seq and Attention models must have bos and eos symbols in voc and slot tag set)')
    parser.add_argument('--beos',action='store_true',help='whether use <BOS> and <EOS> symbol(Seq2Seq and Attention models must set true)')
    #parser.add_argument('-cws,'--context_window_size',default=0,type=int)
    parser.add_argument('--cross',type=int,default=1,help='k-fold cross validation, default k=1, not use cross validation')
    parser.add_argument('-dev','--dev_size',type=float,default=0.33,help='propotion of dev dataset in trainingset')
    parser.add_argument('-bs','--batch_size',type=int,default=64,help='size of mini-batches')
    parser.add_argument('-ep','--epoch',type=int,default=100,help='how many epochs to run')
    parser.add_argument('-es','--early_stop',type=int,default=5,help='maximum patience of dev dataset loss starts to grow')
    parser.add_argument('-cg','--clip_grad',type=float,default=0,help='define the max_norm to perform gradient clip')
    parser.add_argument('-voc','--voc_ratio',type=float,default=0.939)
    parser.add_argument('--post_attention',action='store_true')
    parser.add_argument('--debug',action='store_true',help='debug mode in localhost')
    parser.add_argument('--cuda',action='store_true',help='whether use cuda')
    args=parser.parse_args()

    # set debug mode
    if args.debug:
        args.batch_size=64
        args.embedding_size=32
        args.hidden_size=100
        args.epoch=4
    # model choice
    if args.model.lower() not in ['rnn','rnn+crf','seq2seq','attention','focus']:
        raise ValueError('[Error]: unknown model name',args.model.lower())
    if args.model.lower()in ['seq2seq','attention','focus']:
        args.beos=True
        seq2seq=True #用来标记是否需要特殊的<BEOS>标签作为decoder的第一个输入
    else:
        seq2seq=False
    # set cuda
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
    print('Size of vocabulary is',len(voc_dict))
    slot_tag_dict,reverse_slot_tag_dict=load_tagset(PATH_TO_SLOT_TAGSET[args.domain],seq2seq=seq2seq)
    sentence_tag_dict,reverse_sentence_tag_dict=load_tagset(PATH_TO_SENTENCE_TAGSET[args.domain])
    domain_ontology=Ontology(PATH_TO_ONTOLOGY[args.domain])
    # prepare training and dev dataset
    if args.train or args.validation:
        trainingset=load_trainingdata(PATH_TO_TRAINING_DATA[args.domain],voc_dict,slot_tag_dict,sentence_tag_dict,args.beos,args.debug)
        dataloader=Dataset(trainingset,args.cross,args.dev_size) # 封装数据集用来获取train/dev dataset和minibatches
        print('Size of training set is',len(trainingset))
    if args.test:
        testset=load_trainingdata(PATH_TO_TEST_DATA[args.domain],voc_dict,slot_tag_dict,sentence_tag_dict,args.beos)
        testdataloader=Dataset(testset,1,0)
    pad_num=len(voc_dict)
    ignore_slot_index=-100 if args.model.lower() in ['rnn','rnn+crf'] else len(slot_tag_dict)
    print('Prepare resource time:',(time.time()-start_time))

    if args.validation:
        if args.cross>1:
            pass
        elif args.dev_size>0:
            # simply split the data into training and dev dataset to simply evaluate the model and determine early stop epochs
            print('=======================================================')
            print('Start training with dev dataset size:',args.dev_size)
            if args.model.lower()=='rnn':
                dev_net=SeqRNNModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
            elif args.model.lower()=='rnn+crf':
                dev_net=SeqRNNCRFModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
            elif args.model.lower() in ['focus','seq2seq','attention']:
                dev_net=FocusModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
            if args.cuda:
                dev_net.cuda()
            dev_net.train()
            if args.optimizer=='sgd':
                optimizer=optim.SGD(dev_net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,momentum=args.momentum)
            else:
                optimizer=optim.Adam(dev_net.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args.learning_rate,weight_decay=args.weight_decay)
            if args.early_stop>0:
                patience=0
                tmp_epoch=0
                suggest_epoch=0
            losses=[]
            for epoch in range(args.epoch):
                print('Epoch',(epoch+1),':--------------------------------------------')
                if args.early_stop>0:
                    tmp_epoch+=1
                epoch_loss=torch.Tensor([0])
                if args.cuda:
                    epoch_loss=epoch_loss.cuda()
                for step,(data_sample,length_list) in enumerate(dataloader.get_mini_batches(batch_size=args.batch_size,pad_num=pad_num,ignore_slot_index=ignore_slot_index)):
                    dev_net.train()
                    dev_net.zero_grad()
                    _,padded_data,padded_slot_label,sentence_label,_=data_sample
                    if seq2seq:
                        append_slot_tag=Variable(torch.LongTensor([slot_tag_dict['<BEOS>']]*padded_slot_label.size(0)).unsqueeze(dim=1))
                        padded_slot_label=torch.cat([append_slot_tag,padded_slot_label],dim=1) # [batch_size,max_length+1]
                    if args.cuda:
                        padded_data=padded_data.cuda()
                        padded_slot_label=padded_slot_label.cuda()
                        sentence_label=sentence_label.cuda()
                    # network forward
                    # return form: [batch_size,max_sequence_length,len(slot_tag_dict)],[batch_size,len(sentence_tag_dict)]
                    if args.model.lower()=='rnn':
                        slot_scores,sentence_scores=dev_net(padded_data,length_list)
                        # calculate loss: slot loss plus sentence classification loss
                        slot_loss=slot_loss_function(slot_scores,padded_slot_label,ignore_slot_index=ignore_slot_index,cuda=args.cuda)
                        sentence_loss=sentence_loss_function(sentence_scores,sentence_label)
                        loss=slot_loss+sentence_loss
                    elif args.model.lower()=='rnn+crf':
                        loss=dev_net.cross_entropy_loss(padded_data,length_list,padded_slot_label,sentence_label)
                    elif args.model.lower() in ['seq2seq','focus','attention']:
                        if random.random()+1e-8>0.5:
                            # teacher force learning
                            slot_scores,sentence_scores=dev_net.teacher_force_training(padded_data,length_list,padded_slot_label[:,:-1])
                        else:
                            _,slot_scores,_,sentence_scores=dev_net.decoder_greed(padded_data,length_list,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                        slot_loss=slot_loss_function(slot_scores,padded_slot_label[:,1:],ignore_slot_index=ignore_slot_index,cuda=args.cuda)
                        sentence_loss=sentence_loss_function(sentence_scores,sentence_label)
                        loss=slot_loss+sentence_loss

                    print('\tStep',step,' | mini-batch size',padded_data.size(0),' | temp loss',loss.data[0])
                    # optimize
                    loss.backward()
                    if args.clip_grad>0+1e-6: #梯度截断，防止梯度爆炸
                        torch.nn.utils.clip_grad_norm(dev_net.parameters(),max_norm=args.clip_grad,norm_type=2)
                    optimizer.step()
                    epoch_loss+=loss.data
                print('Total loss for epoch',(epoch+1),'is',epoch_loss[0])
                losses.append(epoch_loss[0])
                # 如果使用early stop, 确定合适的epoch
                if args.early_stop>0:
                    dev_net.eval()
                    total_loss=torch.Tensor([0])
                    if args.cuda:
                        total_loss=total_loss.cuda()
                    # prepare dev dataset
                    for step,(dev_data_sample,dev_length_list) in enumerate(dataloader.get_mini_batches(batch_size=args.batch_size,data=dataloader.dev_data,pad_num=pad_num,ignore_slot_index=ignore_slot_index,shuffle=False)):
                        _,padded_dev_data,padded_dev_slot_label,dev_sentence_label,_=dev_data_sample
                        if seq2seq:
                            append_slot_tag=Variable(torch.LongTensor([slot_tag_dict['<BEOS>']]*padded_dev_slot_label.size(0)).unsqueeze(dim=1))
                            padded_dev_slot_label=torch.cat([append_slot_tag,padded_dev_slot_label],dim=1) # [batch_size,max_length+1]
                        if args.cuda:
                            padded_dev_data=padded_dev_data.cuda()
                            padded_dev_slot_label=padded_dev_slot_label.cuda()
                            dev_sentence_label=dev_sentence_label.cuda()
                        if args.model.lower()=='rnn':    
                            slot_scores,sentence_scores=dev_net(padded_dev_data,dev_length_list)
                            slot_loss=slot_loss_function(slot_scores,padded_dev_slot_label,ignore_slot_index=ignore_slot_index,cuda=args.cuda)
                            sentence_loss=sentence_loss_function(sentence_scores,dev_sentence_label)
                            dev_loss=slot_loss+sentence_loss
                        elif args.model.lower()=='rnn+crf':
                            dev_loss=dev_net.cross_entropy_loss(padded_dev_data,dev_length_list,padded_dev_slot_label,dev_sentence_label)
                        elif args.model.lower() in ['seq2seq','focus','attention']:
                            # totally use teacher force learning
                            slot_scores,sentence_scores=dev_net.teacher_force_training(padded_dev_data,dev_length_list,padded_dev_slot_label[:,:-1])
                            # _,slot_scores,_,sentence_scores=dev_net.decoder_greed(padded_dev_data,dev_length_list,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                            slot_loss=slot_loss_function(slot_scores,padded_dev_slot_label[:,1:],ignore_slot_index=ignore_slot_index,cuda=args.cuda)
                            sentence_loss=sentence_loss_function(sentence_scores,dev_sentence_label)
                            dev_loss=slot_loss+sentence_loss

                        total_loss=total_loss+dev_loss.data
                    if epoch==0:
                        prev_loss=total_loss
                        suggest_epoch=tmp_epoch
                    else:
                        if prev_loss[0]<total_loss[0]:
                            patience+=1
                        else:
                            patience=0
                            suggest_epoch=tmp_epoch
                            prev_loss=total_loss
                            # save model
                            dev_net.save_module(os.path.join(PATH_TO_MODELS[args.domain],'dev.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))
                        if patience>=args.early_stop:
                            print('[Attention]: Early stop works, we stop training after epoch',suggest_epoch)
                            break
            print('=======================================================')
            print('Loss changes during epochs in training:')
            for each_loss in losses:
                print(each_loss,end=' ')
            print('=======================================================')

            if args.early_stop>0:
                args.epoch=suggest_epoch

            if args.cuda:
                dev_net.load_module(os.path.join(PATH_TO_MODELS[args.domain],'dev.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))
                dev_net=dev_net.cpu()
                dev_net.save_module(os.path.join(PATH_TO_MODELS[args.domain],'dev.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))

            # evaluation on dev dataset
            dev_net.load_module(os.path.join(PATH_TO_MODELS[args.domain],'dev.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))
            if args.cuda:
                dev_net.cuda()
            dev_net.eval()
            dev_slot_result_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0}) #分别表示真实标签为x的个数，预测标签为x的个数，真实为x且预测为x的个数，真实不为x但预测为x的个数
            dev_sentence_result_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0}) # precision=TP/P,recall=TP/T,fscore=2*precision*recall/(precision+recall)
            dev_slot_level_result_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0})
            dev_result_dict={'Num':0,'TP':0,'SubSet':0,'SuperSet':0} #'TP'表示完全解析正确,'SubSet'表示解析的slot是原slot的子集，'SuperSet'表示解析的slot是原slot的超集
            file_path=open(os.path.join(PATH_TO_MODELS[args.domain],'dev.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.result'),'w')
            error_fp=open(os.path.join(PATH_TO_MODELS[args.domain],'dev.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.error'),'w')
            for step,(data_sample,dev_length_list) in enumerate(dataloader.get_mini_batches(batch_size=args.batch_size,data=dataloader.dev_data,pad_num=pad_num,ignore_slot_index=ignore_slot_index,shuffle=False)):
                dev_raw_data,padded_dev_data,padded_dev_slot_label,dev_sentence_label,dev_semantics=data_sample
                if seq2seq:
                    append_slot_tag=Variable(torch.LongTensor([slot_tag_dict['<BEOS>']]*padded_dev_slot_label.size(0)).unsqueeze(dim=1))
                    padded_dev_slot_label=torch.cat([append_slot_tag,padded_dev_slot_label],dim=1) # [batch_size,max_length+1]
                if args.cuda:
                    padded_dev_data=padded_dev_data.cuda()
                    padded_dev_slot_label=padded_dev_slot_label.cuda()
                    dev_sentence_label=dev_sentence_label.cuda()
                if args.model.lower()=='rnn':
                    dev_slot_scores,dev_sentence_scores=dev_net(padded_dev_data,dev_length_list)
                    dev_slot_result,dev_sentence_result=bio_decoder(dev_slot_scores,dev_sentence_scores,dev_length_list,reverse_slot_tag_dict,kbest=1,ignore_slot_index=ignore_slot_index)
                    flag=True
                elif args.model.lower()=='rnn+crf':
                    dev_slot_result,dev_sentence_result=dev_net(padded_dev_data,dev_length_list,ignore_slot_index=ignore_slot_index)
                    dev_slot_result,dev_sentence_result=dev_slot_result.data.cpu(),dev_sentence_result.data.cpu()
                    flag=False
                elif args.model.lower() in ['seq2seq','focus','attention']:
                    dev_slot_result,_,_,dev_sentence_result=dev_net.decoder_greed(padded_dev_data,dev_length_list,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                    _,dev_sentence_result=dev_sentence_result.max(dim=1)
                    padded_dev_slot_label,flag=padded_dev_slot_label[:,1:],False
                    # dev_slot_result,_,_,dev_sentence_result=dev_net.decoder_beamer(padded_dev_data,dev_length_list,3,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                    # _,dev_sentence_result=dev_sentence_result.topk(3,dim=1)
                    # padded_dev_slot_label,flag=padded_dev_slot_label[:,1:],True
                    dev_slot_result,dev_sentence_result=dev_slot_result.data.cpu(),dev_sentence_result.data.cpu()
                    
                dev_slot_result_dict,dev_sentence_result_dict=evaluation_char_level(dev_slot_result,dev_sentence_result,
                        padded_dev_slot_label.data,dev_sentence_label.data,reverse_slot_tag_dict,reverse_sentence_tag_dict,
                        kbest=flag,use_ontology=False,slot_dict=dev_slot_result_dict,sentence_dict=dev_sentence_result_dict,ignore_slot_index=ignore_slot_index)
                dev_slot_level_result_dict,dev_result_dict=evaluation_slot_level(dev_slot_result,dev_sentence_result,
                        dev_raw_data,dev_semantics,reverse_slot_tag_dict,reverse_sentence_tag_dict,domain_ontology,fp=file_path,error_fp=error_fp,
                        kbest=flag,use_ontology=False,slot_dict=dev_slot_level_result_dict,sentence_dict=dev_result_dict,ignore_slot_index=ignore_slot_index)    
            print_char_level_scores(dev_slot_result_dict,dev_sentence_result_dict,path=os.path.join(PATH_TO_MODELS[args.domain],'dev.char.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.score'))
            print_slot_level_scores(dev_slot_level_result_dict,dev_result_dict,path=os.path.join(PATH_TO_MODELS[args.domain],'dev.slot.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.score'))
            file_path.close()
            error_fp.close()
        else:
            print('[Warining]:please specify dev dataset size or cross size!')
    
    if args.train:
        print('=======================================================')
        print('Start training using entire dataset:... ...')
        # train the model,use entire dataset and save the net
        if args.model.lower()=='rnn':
            train_net=SeqRNNModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
        elif args.model.lower()=='rnn+crf':
            train_net=SeqRNNCRFModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
        elif args.model.lower() in ['focus','seq2seq','attention']:
            train_net=FocusModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
        train_net.train()
        if args.cuda:
            train_net.cuda()
        if args.optimizer:
            optimizer=optim.SGD(train_net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,momentum=args.momentum)
        else:
            optimizer=optim.Adam(train_net.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=args.learning_rate,weight_decay=args.weight_decay)
        losses=[]
        for epoch in range(args.epoch):
            print('Epoch',(epoch+1),':--------------------------------------------')
            epoch_loss=torch.Tensor([0])
            if args.cuda:
                epoch_loss=epoch_loss.cuda()
            for step,(data_sample,length_list) in enumerate(dataloader.get_mini_batches(batch_size=args.batch_size,data=trainingset,pad_num=pad_num,ignore_slot_index=ignore_slot_index)):
                train_net.zero_grad()
                _,padded_data,padded_slot_label,sentence_label,_=data_sample
                if seq2seq:
                    append_slot_tag=Variable(torch.LongTensor([slot_tag_dict['<BEOS>']]*padded_slot_label.size(0)).unsqueeze(dim=1))
                    padded_slot_label=torch.cat([append_slot_tag,padded_slot_label],dim=1) # [batch_size,max_length+1]
                if args.cuda:
                    padded_data=padded_data.cuda()
                    padded_slot_label=padded_slot_label.cuda()
                    sentence_label=sentence_label.cuda()
                if args.model.lower()=='rnn':
                    slot_scores,sentence_scores=train_net(padded_data,length_list)
                    slot_loss=slot_loss_function(slot_scores,padded_slot_label,ignore_slot_index=ignore_slot_index,cuda=args.cuda)
                    sentence_loss=sentence_loss_function(sentence_scores,sentence_label)
                    loss=slot_loss+sentence_loss
                elif args.model.lower()=='rnn+crf':
                    loss=train_net.cross_entropy_loss(padded_data,length_list,padded_slot_label,sentence_label)
                elif args.model.lower()in ['seq2seq','focus','attention']:
                    if random.random()+1e-8>0.5:
                        # teacher force learning
                        slot_scores,sentence_scores=train_net.teacher_force_training(padded_data,length_list,padded_slot_label[:,:-1])
                    else:
                        _,slot_scores,_,sentence_scores=train_net.decoder_greed(padded_data,length_list,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                    slot_loss=slot_loss_function(slot_scores,padded_slot_label[:,1:],ignore_slot_index=ignore_slot_index,cuda=args.cuda)
                    sentence_loss=sentence_loss_function(sentence_scores,sentence_label)
                    loss=slot_loss+sentence_loss

                print('\tStep',step,' | mini-batch size',padded_data.size(0),' | temp loss',loss.data[0])
                # optimize
                loss.backward()
                if args.clip_grad>0+1e-6: #梯度截断，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm(train_net.parameters(),max_norm=args.clip_grad,norm_type=2)
                optimizer.step()
                epoch_loss+=loss.data
            print('Total loss for epoch',(epoch+1),'is',epoch_loss[0])
            losses.append(epoch_loss[0])
        print('=======================================================')
        print('Loss changes during epochs in training:')
        for each_loss in losses:
            print(each_loss,end=' ')
        print('=======================================================')
        print('Training finished!')
        if args.cuda:
            train_net=train_net.cpu()
        train_net.save_module(os.path.join(PATH_TO_MODELS[args.domain],'train.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))

    if args.test:
        # testing process
        if args.train:
            test_net=train_net
        else:
            # load model
            if args.model.lower()=='rnn':
                test_net=SeqRNNModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
            elif args.model.lower()=='rnn+crf':
                test_net=SeqRNNCRFModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
            elif args.model.lower() in ['focus','seq2seq','attention']:
                test_net=FocusModel(args,len(voc_dict),len(slot_tag_dict),len(sentence_tag_dict))
            test_net.load_module(os.path.join(PATH_TO_MODELS[args.domain],'train.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.model'))
        if args.cuda:
            test_net=test_net.cuda()
        test_net.eval()
        # forward and decoder
        test_slot_result_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0}) #分别表示真实标签为x的个数，预测标签为x的个数，真实为x且预测为x的个数，真实不为x但预测为x的个数
        test_sentence_result_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0}) # precision=TP/P,recall=TP/T,fscore=2*precision*recall/(precision+recall)
        test_slot_level_result_dict=defaultdict(lambda :{'T':0,'P':0,'TP':0,'FP':0})
        test_result_dict={'Num':0,'TP':0,'SubSet':0,'SuperSet':0} #'TP'表示完全解析正确,'SubSet'表示解析的slot是原slot的子集，'SuperSet'表示解析的slot是原slot的超集
        file_path=open(os.path.join(PATH_TO_MODELS[args.domain],'test.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.result'),'w')
        error_fp=open(os.path.join(PATH_TO_MODELS[args.domain],'test.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.error'),'w')
        post_processor=PostProcessor(PATH_TO_CONSTRAINT_MAPPINGS[args.domain],PATH_TO_POST_PROCESS_DIR[args.domain],domain_ontology)
        for step,(data_sample,test_length_list) in enumerate(testdataloader.get_mini_batches(batch_size=args.batch_size,data=testdataloader.data,pad_num=pad_num,ignore_slot_index=ignore_slot_index,shuffle=False)):
            test_raw_data,padded_test_data,padded_test_slot_label,test_sentence_label,test_semantics=data_sample
            if seq2seq:
                append_slot_tag=Variable(torch.LongTensor([slot_tag_dict['<BEOS>']]*padded_test_slot_label.size(0)).unsqueeze(dim=1))
                padded_test_slot_label=torch.cat([append_slot_tag,padded_test_slot_label],dim=1) # [batch_size,max_length+1]
            if args.cuda:
                padded_test_data=padded_test_data.cuda()
                padded_test_slot_label=padded_test_slot_label.cuda()
                test_sentence_label=test_sentence_label.cuda()
            if args.model.lower()=='rnn':
                test_slot_scores,test_sentence_scores=test_net(padded_test_data,test_length_list)
                # test_slot_result,test_sentence_result=kbest_decoder(test_slot_scores,test_sentence_scores,test_length_list,kbest=1,ignore_slot_index=ignore_slot_index)
                test_slot_result,test_sentence_result=bio_decoder(test_slot_scores,test_sentence_scores,test_length_list,reverse_slot_tag_dict,kbest=1,ignore_slot_index=ignore_slot_index)
                flag=True
            elif args.model.lower()=='rnn+crf':
                test_slot_result,test_sentence_result=test_net(padded_test_data,test_length_list,ignore_slot_index=ignore_slot_index)
                test_slot_result,test_sentence_result=test_slot_result.data.cpu(),test_sentence_result.data.cpu()
                flag=False
            elif args.model.lower() in ['seq2seq','focus','attention']:
                test_slot_result,_,_,test_sentence_result=test_net.decoder_greed(padded_test_data,test_length_list,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                _,test_sentence_result=test_sentence_result.max(dim=1)
                padded_test_slot_label,flag=padded_test_slot_label[:,1:],False
                # test_slot_result,_,_,test_sentence_result=test_net.decoder_beamer(padded_test_data,test_length_list,3,init_tags=slot_tag_dict['<BEOS>'],ignore_slot_index=ignore_slot_index)
                # _,test_sentence_result=test_sentence_result.topk(3,dim=1)
                # padded_test_slot_label,flag=padded_test_slot_label[:,1:],True
                test_slot_result,test_sentence_result=test_slot_result.data.cpu(),test_sentence_result.data.cpu()
            
            test_slot_result_dict,test_sentence_result_dict=evaluation_char_level(test_slot_result,test_sentence_result,
                    padded_test_slot_label.data,test_sentence_label.data,reverse_slot_tag_dict,reverse_sentence_tag_dict,
                    kbest=flag,use_ontology=False,slot_dict=test_slot_result_dict,sentence_dict=test_sentence_result_dict,ignore_slot_index=ignore_slot_index)
            test_slot_level_result_dict,test_result_dict=evaluation_slot_level(test_slot_result,test_sentence_result,
                    test_raw_data,test_semantics,reverse_slot_tag_dict,reverse_sentence_tag_dict,domain_ontology,fp=file_path,error_fp=error_fp,
                    kbest=flag,use_ontology=False,slot_dict=test_slot_level_result_dict,sentence_dict=test_result_dict,ignore_slot_index=ignore_slot_index,post_processor=post_processor)                 
        print_char_level_scores(test_slot_result_dict,test_sentence_result_dict,path=os.path.join(PATH_TO_MODELS[args.domain],'test.char.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.score'))
        print_slot_level_scores(test_slot_level_result_dict,test_result_dict,path=os.path.join(PATH_TO_MODELS[args.domain],'test.slot.'+args.model.lower()+'.'+args.cell+'.'+str(args.embedding_size)+'.'+str(args.hidden_size)+'.score'))
        file_path.close()
        error_fp.close()

if __name__=='__main__':

    main(sys.argv)