import numpy as np
from sklearn.metrics import accuracy_score
import copy
import logging
import torch
import time
from tqdm import trange, tqdm
import torch.nn as nn
from utils.utils import flatten_model


def test_local(args):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    (id, pid, model, client, local_model_weight, local_loss) = args
    model = model.to(device)
    local_model = copy.deepcopy(model).to(device)
    # optimizer = torch.optim.SGD(local_model.parameters(), lr=client.lr)
    criterion = nn.CrossEntropyLoss()
    train_dataloader = client.train_dataloader
    train_loss = 0
    ep_loss = 0
    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        output = local_model(X)
        loss = criterion(output, y)
        train_loss += loss.item()

    ep_loss = train_loss / len(train_dataloader)
    local_loss[id] = ep_loss


def train(args):
    
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    (id, pid, model, client, local_model_weight,
     train_local_loss, local_inference_loss, training_time, algorithm) = args
    # model = model.cuda()
    model = model.to(device)
    train_dataloader = client.train_dataloader
    # local_model = copy.deepcopy(model).cuda()
    _, start_inference_loss = test(model, train_dataloader)
    local_inference_loss[id, 0] = start_inference_loss
    local_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=client.lr)
    criterion = nn.CrossEntropyLoss()
    t = time.time()
    local_model.train()

    for i in range(client.eps):
        ep_loss = 0
        train_loss = 0.0
        for X, y in train_dataloader:
            print(len(train_dataloader))
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = local_model(X)
            if algorithm == "FedProx":
                proximal_term = torch.tensor(0.).to(device)
                for w, w_t in zip(model.parameters(), local_model.parameters()):
                    proximal_term += torch.pow(torch.norm(w - w_t), 2)
                loss = criterion(output, y) + proximal_term * client.mu / 2
            else:
                loss = criterion(output, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        ep_loss += train_loss / len(train_dataloader)
        # print(f"Client : {pid} Number sample : {client.n_samples} Epoch : {i}   Ep loss : {train_loss/len(train_dataloader)}")
        train_local_loss[id, i] = ep_loss

    _, final_inference_loss = test(local_model, train_dataloader)
    local_inference_loss[id, 1] = final_inference_loss
    training_time[id,0] = time.time() - t
    local_model_weight[id] = flatten_model(local_model)


def test(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    y_prd = []
    y_grt = []
    cel = nn.CrossEntropyLoss()
    loss = 0.0
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss += cel(output, y).item()
        output = output.argmax(-1)
        y_prd.append(output.cpu())
        y_grt.append(y.cpu())
    loss = loss/len(test_dataloader)

    y_ = np.concatenate([i.numpy() for i in y_prd])
    y_gt = np.concatenate([i.numpy() for i in y_grt])

    return accuracy_score(y_, y_gt), loss
