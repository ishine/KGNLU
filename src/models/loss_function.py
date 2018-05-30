#!/usr/bin/env python3
#coding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable

def slot_loss_function(slot_scores,slot_tags,ignore_slot_index=-100,cuda=False):
    '''  
    Arguments:
        slot_scores : [batch_size,max_sequence_length,len(slot_dict)]
        slot_tags : [batch_size,max_sequence_length of slot tag index]
        ignore_slot_index : padding index in slot_tags
    '''

    loss=nn.NLLLoss(ignore_index=ignore_slot_index)
    # slot_tag_length=slot_scores.size(2)
    # loss_scores=loss(slot_scores.contiguous().view(-1,slot_tag_length),slot_tags.view(-1))
    # return loss_scores
    total_score=Variable(torch.Tensor([0]))
    if cuda:
        total_score=total_score.cuda()
    for each in range(slot_scores.size(0)):
        total_score=total_score+loss(slot_scores[each],slot_tags[each])
    return total_score/slot_scores.size(0)

def sentence_loss_function(sentence_score,sentence_tags):
    loss=nn.NLLLoss()
    loss_scores=loss(sentence_score,sentence_tags)
    return loss_scores
