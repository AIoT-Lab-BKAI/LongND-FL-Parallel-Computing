import copy
import logging
import torch
import time
from tqdm import trange, tqdm
import torch.nn as nn
from utils.utils import flatten_model


def train(args):
    (id, pid, model, client, local_model_weight, train_local_loss,algorithm) = args
    device = torch.device("cuda")
    model = model.to(device)
    local_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=client.lr)
    criterion = nn.CrossEntropyLoss()
    t = time.time()
    local_model.train()

    for i in range(client.eps):
        
        ep_loss = 0
        train_dataloader = client.train_dataloader
        train_loss = 0.0
        for X, y in tqdm(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = local_model(X)
            
            if algorithm == "fedprox":
                proximal_term = 0.0
                # breakpoint()
                for w, w_t in zip(local_model.parameters(), model.parameters()):
                    proximal_term += (w-w_t).norm(1)
                loss = criterion(output, y) + (proximal_term**2) * client.mu/2
            else:
                loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        ep_loss += train_loss / len(train_dataloader)
        print(
            f"Client : {pid} Number sample : {client.n_samples} Epoch : {i}   Ep loss : {train_loss/len(train_dataloader)}"
        )
        train_local_loss[id, i] = ep_loss
    local_model_weight[id] = flatten_model(local_model)


from sklearn.metrics import accuracy_score
import numpy as np


def test(model, test_dataloader,device):
    print("Test :-------------------------------")
    y_prd = []
    y_grt = []
    cel = nn.CrossEntropyLoss()
    loss = 0.0
    for X, y in tqdm(test_dataloader):
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss += cel(output, y).item()
        output = output.argmax(-1)
        y_prd.append(output)
        y_grt.append(y)
    loss = loss/len(test_dataloader)
    y_ = np.concatenate([i.numpy() for i in y_prd])
    y_gt = np.concatenate([i.numpy() for i in y_grt])
    return accuracy_score(y_, y_gt), loss

