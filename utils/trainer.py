import numpy as np
from sklearn.metrics import accuracy_score
import copy
import logging
import torch
import time
from tqdm import trange, tqdm
import torch.nn as nn
from utils.utils import flatten_model


def train(args):
    (id, pid, model, client, local_model_weight, train_local_loss, algorithm) = args
    model = model.cuda()
    local_model = copy.deepcopy(model).cuda()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=client.lr)
    criterion = nn.CrossEntropyLoss()
    t = time.time()
    local_model.train()

    for i in range(client.eps):
        ep_loss = 0
        train_dataloader = client.train_dataloader
        train_loss = 0.0
        for X, y in train_dataloader:
            X = X.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            output = local_model(X)

            if algorithm == "fedprox":
                proximal_term = torch.tensor(0.).cuda()
                for w, w_t in zip(model.parameters(), local_model.parameters()):
                    proximal_term += torch.pow(torch.norm(w - w_t), 2)
                loss = criterion(output, y) + proximal_term * client.mu / 2
            else:
                loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        ep_loss += train_loss / len(train_dataloader)
        # print(
        #     f"Client : {pid} Number sample : {client.n_samples} Epoch : {i}   Ep loss : {train_loss/len(train_dataloader)}")
        train_local_loss[id, i] = ep_loss
    local_model_weight[id] = flatten_model(local_model)


def test(model, test_dataloader):
    model = model.cuda()
    y_prd = []
    y_grt = []
    cel = nn.CrossEntropyLoss()
    loss = 0.0
    for X, y in test_dataloader:
        X = X.cuda()
        y = y.cuda()
        output = model(X)
        loss += cel(output, y).item()
        output = output.argmax(-1)
        y_prd.append(output.cpu())
        y_grt.append(y.cpu())
    loss = loss/len(test_dataloader)

    y_ = np.concatenate([i.numpy() for i in y_prd])
    y_gt = np.concatenate([i.numpy() for i in y_grt])

    return accuracy_score(y_, y_gt), loss
