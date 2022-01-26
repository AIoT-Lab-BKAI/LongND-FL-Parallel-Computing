import json
import functools
from re import T
import torch.distributed as dist
import sys
import math
import logging
import collections
import copy
from random import random
from collections import OrderedDict
import numpy as np
from numpy.core.defchararray import count
# from torch.cuda.memory import reset_accumulated_memory_stats
import torch
import torch.nn as nn
import wandb


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
    train_client = list(set(list_cl_per_round) - set(drop_client))
    return drop_client, train_client


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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
        output.append(flat[count: count + n].view_as(tensor))
        count += n
    output = tuple(output)
    temp = OrderedDict()
    for i, j in enumerate(model.state_dict().keys()):
        temp[j] = output[i]
    return temp


def load_epoch(list_client, list_epochs):
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


def standardize_weights(dqn_weights, n_models):
    s_func = nn.Softmax(dim=0)
    means = [dqn_weights[0, cli*2] for cli in range(n_models)]
    s_means = s_func(torch.FloatTensor(means))
    s_std = torch.Tensor([np.clip(dqn_weights[0, cli*2+1]/100, 0.001, s_means[cli] * 0.2) for cli in range(n_models)])
    s_epochs = s_std
    assigned_priorities = torch.Tensor([np.random.normal(s_means[i], s_std[i]) for i in range(n_models)])
    
    return s_means.numpy(), s_std.numpy(), s_epochs.numpy(), assigned_priorities.numpy()


def aggregate(local_weight, n_models, assigned_priorities):
    ratio = torch.Tensor(np.array(assigned_priorities))
    return torch.squeeze(ratio.t() @ local_weight)


def aggregate_benchmark(local_weight, n_models):
    ratio = torch.ones(1,n_models)/n_models
    return torch.squeeze(ratio @ local_weight)


def aggregate_benchmark_fedadp(local_weight, global_weight, train_clients, smooth_angle, round):
    """
    :param local_weights the weights of model after SGD updates
    :param global_weight the weight of the global model
    :param train_clients the list contain all clients trained in this round
    """

    model_difference = local_weight.to('cpu') - global_weight.to('cpu')

    F_i = - model_difference / 0.01

    D_i = [len(client.train_dataloader) for client in train_clients]
    D_i = torch.FloatTensor(D_i / np.sum(D_i))

    F = D_i.T @ F_i

    corel = F.unsqueeze(0) @ F_i.T

    corel_norm = torch.clip(corel / (torch.norm(F_i) * torch.norm(F)), min=-1, max=1)
    instantaneous_angle = torch.squeeze(torch.arccos(corel_norm))

    # with open("corel.txt", "a+") as file:
    #     file.write(str(list(corel.detach().numpy())) + "\n")
    print(list(corel.detach().numpy()))

    # with open("corel_norm.txt", "a+") as file:
    #     file.write(str(list(corel_norm.detach().numpy())) + "\n")
    print(list(corel_norm.detach().numpy()))

    if (smooth_angle is None):
        smooth_angle = instantaneous_angle
    else:
        smooth_angle = (round - 1)/round * smooth_angle + 1/round * instantaneous_angle
    
    with open("smooth_angle.txt", "a+") as file:
        file.write(str(list(smooth_angle.detach().numpy())) + "\n")

    impact_factor = torch.squeeze(5 * (1 - torch.exp( - torch.exp(- 5 * (smooth_angle - torch.ones_like(smooth_angle))))))

    with open("impact_factor.txt", "a+") as file:
        file.write(str(list(impact_factor.detach().numpy())) + "\n")

    normalized_impact_factor = torch.exp(impact_factor)/torch.sum(torch.exp(impact_factor))

    return torch.squeeze(normalized_impact_factor.T @ local_weight), smooth_angle


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


def save_dataset_idx(list_idx_sample, path="dataset_idx.json"):
    with open(path, "w+") as outfile:
        json.dump(list_idx_sample, outfile)


def load_dataset_idx(path="data"):
    list_idx = json.load(open(path, 'r'))
    return {int(k): v for k, v in list_idx.items()}


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


def getDictionaryLosses(start_loss, final_loss, num_clients):
    client_dicts = {}
    for cli in range(num_clients):
        cli_dict = {}
        cli_dict["local_inference_loss"] = final_loss[cli]
        cli_dict["server_inference_loss"] = start_loss[cli]
        client_dicts[cli] = cli_dict
    return client_dicts


def get_mean_losses(local_train_losses, num_cli):
    num_epoches = torch.count_nonzero(local_train_losses, dim = 1)
    final_losses = [local_train_losses[i, num_epoches[i]-1] for i in range(num_cli)]
    start_losses = [local_train_losses[i, 0]
                    for i in range(num_cli)]
    means = [torch.sum(local_train_losses[i, ]) /
             num_epoches[i] for i in range(num_cli)]
    stds = [torch.std(local_train_losses[i, num_epoches[i]], unbiased=False) for i in range(num_cli)]
    return start_losses, final_losses, stds
