from random import random
from collections import OrderedDict
import numpy as np
from numpy.core.defchararray import count
from torch.cuda.memory import reset_accumulated_memory_stats
import torch
import torch.nn as nn


def GenerateLocalEpochs(percentage, size, max_epochs):
    """ Method generates list of epochs for selected clients
  to replicate system heteroggeneity

  Params:
    percentage: percentage of clients to have fewer than E epochs
    size:       total size of the list
    max_epochs: maximum value for local epochs
  
  Returns:
    List of size epochs for each Client Update

  """

    # if percentage is 0 then each client runs for E epochs
    if percentage == 0:
        return np.array([max_epochs] * size)
    else:
        # get the number of clients to have fewer than E epochs
        heterogenous_size = int((percentage / 100) * size)

        # generate random uniform epochs of heterogenous size between 1 and E
        epoch_list = np.random.randint(1, max_epochs, heterogenous_size)

        # the rest of the clients will have E epochs
        remaining_size = size - heterogenous_size
        rem_list = [max_epochs] * remaining_size
        epoch_list = np.append(epoch_list, rem_list, axis=0)
        # shuffle the list and return
        np.random.shuffle(epoch_list)

        return epoch_list


import copy
import torch


def weight_avg(list_model_state):
    weights_avg = copy.deepcopy(list_model_state[0])
    for k in weights_avg.keys():
        for i in range(1, len(list_model_state)):
            weights_avg[k] += list_model_state[i][k]

        weights_avg[k] = torch.div(weights_avg[k], len(list_model_state))

    global_weights = weights_avg
    return global_weights


def select_client(n_clients, n_cl_per_round):
    return np.random.choice(range(n_clients), n_cl_per_round, replace=False)


def select_drop_client(list_cl_per_round, drop_percent):
    n_cl = len(list_cl_per_round)
    import math

    n_drop = max(math.floor(n_cl * drop_percent), 1)
    # Dung test for use all clients at a time
    if drop_percent == 0.0:
        n_drop = 0
    drop_client = np.random.choice(list_cl_per_round, n_drop)
    train_clinet = list(set(list_cl_per_round) - set(drop_client))
    return drop_client, train_clinet


def count_params(model):
    return sum(p.numel() for p in model.parameters())


import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist
import functools


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def flatten_model(model):
    ten = torch.cat([flatten_tensors(i) for i in model.parameters()])
    return ten


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def unflatten_model(flat, model):
    count = 0
    l = []
    output = []
    for tensor in model.parameters():
        n = tensor.numel()
        output.append(flat[count : count + n].view_as(tensor))
        count += n
    output = tuple(output)
    temp = OrderedDict()
    for i, j in enumerate(model.state_dict().keys()):
        temp[j] = output[i]
    # retrun temp
    return temp

def load_epoch(list_client,list_epochs):
    n_client = len(list_client)
    for i in range(n_client):
        list_client[i].eps = list_epochs[i]

def communicate(tensors, communication_op):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)


import torch

def standardize_weights(dqn_weights, n_models):
    s_func = nn.Softmax(dim=0)
    means = [dqn_weights[0, cli*3] for cli in range(n_models)]
    s_means = s_func(torch.FloatTensor(means))
    s_std = [dqn_weights[0, cli*3+1]/100 for cli in range(n_models)]
    s_epochs = [math.ceil(dqn_weights[0,cli*3+1]*10) if math.ceil(dqn_weights[0,cli*3+1]*10) > 0 else 1 for cli in range(n_models)]
    assigned_priorities = [np.random.normal(s_means[i], s_std[i]) for i in range(n_models)]
    return s_means, s_std, s_epochs, assigned_priorities

def aggregate(local_weight, n_models, assigned_priorities):
    # weighted_ratio = []
    # _, _, _, assigned_priorities = standardize_weights(dqn_weights, n_models)

    # for cli in range(0, n_models):
    #     weighted_ratio.append(np.random.normal(dqn_weights[0, cli*2], dqn_weights[0, cli*2+1], 1))
    ratio = torch.Tensor(np.array(assigned_priorities))
    # ratio = assigned_prorities
    # ratio = torch.ones(1,n_models)/n_models
    # print(ratio.shape)
    # print(local_weight.shape)
    # return torch.squeeze(ratio @ local_weight.t())
    return torch.squeeze(ratio.t() @ local_weight)


def generate_abiprocess(mu, sigma, n_client):
    s = np.random.normal(mu, sigma, n_client)
    with open("abi_process.txt", "w") as f:
        for item in s:
            f.write(f"{item}\n")
        f.close()
    # return s


def read_abiprocesss(path="abi_process.txt"):
    list_abipro = []
    with open(path, "r") as f:
        for i in f:
            list_abipro.append(float(i))
        f.close()
    return list_abipro


def convert_tensor_to_list(train_local_loss):
    result = []
    for i in train_local_loss:
        result.append([float(j) for j in i])
    return result


import json


def save_infor(list_sam, path="sample.json"):
    with open(path, "w+") as outfile:
        json.dump(list_sam, outfile)

def log_by_round(sample, path="samples.json"):
    with open(path, "a+") as outfile:
        outfile.write(str(sample))
        outfile.write("/n")


def get_train_time(n_sample, list_abiprocess):
    train_time = np.array(n_sample) / np.array(list_abiprocess)
    min_time = np.amin(train_time)
    max_time = np.amax(train_time)
    delay = max_time - min_time
    return train_time, delay, max_time, min_time

def save_dataset_idx(list_idx_sample,path="dataset_idx.json"):
    with open(path, "w+") as outfile:
        json.dump(list_idx_sample, outfile)

def load_dataset_idx(path="data"):
    list_idx =json.load(open(path,'r'))
    return {int(k):v for k,v in list_idx.items()}

def getLoggingDictionary(sample, num_clients):
    client_dicts = {}
    for cli in range(num_clients):
        cli_dict = {}
        cli_dict["mean"] = sample["means"][cli]
        cli_dict["std"] = sample["std"][cli]
        cli_dict["epoch"] = sample["num_epochs"][cli]
        cli_dict["priority"] = sample["assigned_priorities"][cli]
        client_dicts[cli] = cli_dict
    return client_dicts
def getDictionaryLosses(losses, num_clients):
    client_dicts = {}
    for cli in range(num_clients):
        cli_dict = {}
        cli_dict["local_loss"] = losses[cli]
        client_dicts[cli] = cli_dict
    return client_dicts
    
def get_mean_losses(local_train_losses, num_cli):
    return [torch.sum(local_train_losses[i:,])/torch.count_nonzero(local_train_losses[i:,]) for i in range(num_cli)]
# import numpy as np

# if __name__ == "__main__":
#     a = np.array([1,2,3])
#     b = np.array([2,3,4])
#     print(a/b)
