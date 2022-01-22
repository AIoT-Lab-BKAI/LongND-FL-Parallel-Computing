import argparse
from math import ceil
import os.path
from os import path
from os import stat
from re import M
from torchvision import transforms, datasets
from tqdm import tqdm
from utils.loader import CustomDataset
from utils.utils import (
    load_dataset_idx,
)
import random
import numpy as np
import torch
import copy
from utils.trainer import test
from torch.utils.data import DataLoader
from utils.option import option
from models.models import MNIST_CNN, CNNCifar
from models.vgg import vgg11
from ddpg_agent.ddpg import *
import wandb
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(dataset_name, path_data_idx):
    if dataset_name == "mnist":
        transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST("data/mnist/", train=True, download=True, transform=transforms_mnist)
        test_dataset = datasets.MNIST("data/mnist/", train=False, download=True, transform=transforms_mnist)
        list_idx_sample = load_dataset_idx(path_data_idx)

    elif dataset_name == "cifar100":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10("./data/cifar/", train=True, download=True,transform=apply_transform)

        test_dataset = datasets.CIFAR10("./data/cifar/", train=False, download=True,transform=apply_transform)

        list_idx_sample = load_dataset_idx(path_data_idx)
    elif dataset_name == "fashionmnist":
        transforms_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.FashionMNIST("./data/fashionmnist/", train=True, download=True,
                                         transform=transforms_mnist)

        test_dataset = datasets.FashionMNIST("./data/fashionmnist/", train=False, download=True,
                                        transform=transforms_mnist)
        list_idx_sample = load_dataset_idx(path_data_idx)
    else:
        warnings.warn("Dataset not supported")
        exit()

    return train_dataset, test_dataset, list_idx_sample


def init_model(dataset_name):
    if dataset_name == "mnist":
        model = MNIST_CNN()
    elif dataset_name == "cifar100":
        model = vgg11(100)
        # print(model)
    elif dataset_name == "fashionmnist":
        model = MNIST_CNN()
    else:
        warnings.warn("Model not supported")
    return model

def train(model, dataloader,optimizer, criterion, device):
    model.train()
    train_loss =0 
    for (X,y) in dataloader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/ len(dataloader)
    
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    
    train_dataset, test_dataset, list_idx_sample = load_dataset(args.dataset_name, args.path_data_idx)
    model = init_model(args.dataset_name)
    data_idx = list_idx_sample[0]
    model.to(device)
    train_dataset = CustomDataset(train_dataset, data_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
    cel_loss =  nn.CrossEntropyLoss()
    # breakpoint()
    for round in tqdm(range(args.num_rounds)):
        train_loss = train(model, train_dataloader, optimizer, cel_loss, device)
        acc,test_loss = test(model, test_dataloader)
        wandb.log({"loss/train_loss":train_loss, "loss/test_loss":test_loss, "accuracy/acc":acc})
    
if __name__ == '__main__':
    parse_args = option()
    args = parse_args
    wandb.init(project="federated-learning-ICDCS",
               entity="aiotlab",
               name=parse_args.run_name,
               group=parse_args.group_name,
               #    mode="disabled",
               config={
                   "num_rounds": parse_args.num_rounds,
                   "batch_size": parse_args.batch_size,
                   "path_data_idx": parse_args.path_data_idx,
                   "learning_rate": parse_args.learning_rate,
                   "seed": parse_args.seed,
                   "log_dir": parse_args.log_dir,
                   "dataset_name": parse_args.dataset_name,
                   "num_workers": parse_args.num_workers,
               })
    args = wandb.config
    wandb.define_metric("test_acc", summary="max")
    print(">>> START RUNNING: {}  - Dataset: {}".format(parse_args.run_name, args.dataset_name))
    main(args)
    wandb.finish()
