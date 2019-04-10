# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np



class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, options, model_output, y, loss):
        return loss(options, model_output, y, loss)


def loss(options, model_output, y, loss, mode):
    num_support = 0
    num_query = 0
    classes_per_it = 0
    if mode == 'train':
        model_output = model_output.view(options.classes_per_it_tr,(options.num_support_tr+options.num_query_tr)*2,-1).requires_grad_()
        num_support = options.num_support_tr
        num_query = options.num_query_tr
        classes_per_it = options.classes_per_it_tr
    if mode == 'dev':
        model_output = model_output.view(options.classes_per_it_val,(options.num_support_val+options.num_query_val)*2,-1)
        num_support = options.num_support_val
        num_query = options.num_query_val
        classes_per_it = options.classes_per_it_val


    model_ripe,model_rotten = torch.split(model_output, num_support + num_query, dim=1)
    prototypical_ripe,query_ripe = torch.split(model_ripe, list( (num_support, num_query) ), dim=1)
    prototypical_rotten, query_rotten = torch.split(model_rotten, list( (num_support, num_query) ), dim=1)

    prototypical_ripe = torch.mean(prototypical_ripe, 1, True)
    prototypical_rotten = torch.mean(prototypical_rotten, 1, True)

    two_proto_expand = torch.cat((prototypical_ripe, prototypical_rotten), 1)

    query_ripe_list = torch.split(query_ripe, 1, dim=1)
    query_rotten_list = torch.split(query_rotten, 1, dim=1)

    # ripe_mse_tensor = torch.tensor([0,0,0,0,0])
    accurate_num = 0
    # val_tensor = torch.zeros(13)
    for query_ripe in query_ripe_list:
        query_ripe = query_ripe.expand(classes_per_it,2,-1)
        ripe_distance = torch.pow(two_proto_expand-query_ripe,2)
        ripe_distance = torch.sum(ripe_distance, 2)
        # ripe_sum = torch.sum(ripe_distance,1)
        ripe_distance = ripe_distance.float()/torch.sum(ripe_distance,1).view(classes_per_it,1)
        ripe_ripe_dis,ripe_rotten_dis = torch.split(ripe_distance, 1, dim=1)
        dis = ripe_ripe_dis-ripe_rotten_dis
        accurate_num += torch.max(ripe_distance, 1)[1].sum()
        #
        # val_tensor = np.concatenate((val_tensor.numpy(),torch.max(ripe_distance, 1)[1].numpy),0)
        # print(torch.max(ripe_distance, 1)[1])
        #
        loss += dis.sum()

    for query_rotten in query_rotten_list:
        query_rotten = query_rotten.expand(classes_per_it,2,-1)
        rotten_distance = torch.pow(two_proto_expand-query_rotten,2)
        rotten_distance = torch.sum(rotten_distance, 2)
        rotten_distance = rotten_distance.float()/torch.sum(rotten_distance,1).view(classes_per_it,1)
        rotten_ripe_dis, rotten_rotten_dis = torch.split(rotten_distance, 1, dim=1)
        dis = rotten_rotten_dis - rotten_ripe_dis
        accurate_num += torch.min(rotten_distance, 1)[1].sum()
        #
        # val_tensor = np.cat((val_tensor,torch.min(rotten_distance, 1)[1]),0)
        # print(torch.min(rotten_distance, 1)[1])
        #
        loss+= dis.sum()

    # print(torch.sum(val_tensor,1))
    return loss, accurate_num.item()