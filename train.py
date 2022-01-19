import argparse
from math import ceil
import os.path
from os import path
from os import stat
from re import M
from numpy.core.arrayprint import str_format
from numpy.core.defchararray import count
from numpy.lib.function_base import _percentile_dispatcher
from numpy.lib.npyio import save
from torchvision import transforms, datasets
from modules.Client import Client
from tqdm import tqdm
from utils.utils import (
    aggregate_benchmark_fedadp,
    flatten_model,
    get_mean_losses,
    getDictionaryLosses,
    getLoggingDictionary,
    aggregate_benchmark,
    select_client,
    select_drop_client,
    standardize_weights,
    read_abiprocesss,
    generate_abiprocess,
    load_dataset_idx,
    get_train_time,
    count_params,
    aggregate,
    unflatten_model,
    load_epoch,
)
from utils.trainer import train, test_local
import random
import numpy as np
import torch
import torch.multiprocessing as mp
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
        print(model)
    elif dataset_name == "fashionmnist":
        model = MNIST_CNN()
    else:
        warnings.warn("Model not supported")
    return model


def main(args):
    """ Parse command line arguments or load defaults """
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    generate_abiprocess(mu=100, sigma=5, n_client=args.num_clients)
    list_abiprocess_client = read_abiprocesss()
    assert len(list_abiprocess_client) == args.num_clients, "not enough abi-processes"

    # >>>> START: LOAD DATASET & INIT MODEL
    train_dataset, test_dataset, list_idx_sample = load_dataset(args.dataset_name, args.path_data_idx)
    client_model = init_model(args.dataset_name)
    n_params = count_params(client_model)
    prev_reward = None

    list_client = [
        Client(
            idx=idx,
            dataset=train_dataset,
            list_idx_sample=list_idx_sample,
            list_abiprocess=list_abiprocess_client,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            epochs=args.num_epochs,
            mu=args.mu,
        )
        for idx in range(args.num_clients)
    ]

    list_trained_client = []

    # >>>> SERVER: INITIALIZE MODEL
    # This is dimensions' configurations for the DQN agent
    state_dim = args.clients_per_round * 3  # each agent {start_loss, end_loss, } = 30
    # plus action for numbers of epochs for each client
    action_dim = args.clients_per_round * 2 # = 10
    # action_dim = args.clients_per_round * 4  # = 10

    agent = DDPG_Agent(state_dim=state_dim, action_dim=action_dim, log_dir=args.log_dir, beta=args.beta, hidden_dim = args.hidden_dim,
        init_w=args.init_w,
        value_lr=args.value_lr,
        policy_lr=args.policy_lr,
        max_steps=args.max_steps,
        max_frames=args.max_frames,
        batch_size=args.batch_size_ddpg,
        gamma = args.gamma,
        soft_tau = args.soft_tau).to(device)
    # agent = DDPG_Agent(state_dim=state_dim, action_dim=action_dim, log_dir=args.log_dir, beta=args.beta).cuda()

    # Multi-process training
    pool = mp.Pool(args.num_core)
    smooth_angle = None         # Use for fedadp

    for round in tqdm(range(args.num_rounds)):
        # mocking the number of epochs that are assigned for each client.
        dqn_list_epochs = [args.num_epochs for _ in range(args.clients_per_round)]

        # Ngau nhien lua chon client de train
        selected_client = select_client(args.num_clients, args.clients_per_round)
        drop_clients, train_client = select_drop_client(selected_client, args.drop_percent)
        train_clients = list(set(selected_client) - set(drop_clients))

        # Khoi tao cac bien su dung train
        local_model_weight = torch.zeros(len(train_clients), n_params)
        local_model_weight.share_memory_()

        # train_local_loss = torch.zeros(len(train_client), args.num_epochs)
        # maximum number of epochs for client is 100
        train_local_loss = torch.zeros(len(train_client), 100)
        train_local_loss.share_memory_()
        list_trained_client.append(train_clients)
        list_abiprocess = [list_client[i].abiprocess for i in train_clients]
        local_n_sample = np.array([list_client[i].n_samples for i in train_clients]) * \
            np.array([list_client[i].eps for i in train_clients])
        local_inference_loss = torch.zeros(len(train_client), 2)
        local_inference_loss.share_memory_()
        print("ROUND: ", round)
        print([list_client[i].eps for i in train_clients])
        start_l, final_l = 0, 0

        # Huan luyen song song tren cac client
        pool.map(
            train,
            [
                (
                    i,
                    train_clients[i],
                    copy.deepcopy(client_model),
                    list_client[train_clients[i]],
                    local_model_weight,
                    train_local_loss,
                    local_inference_loss,
                    args.algorithm,
                )
                for i in range(len(train_clients))
            ],
        )

        if args.train_mode == "benchmark":
            flat_tensor = aggregate_benchmark(local_model_weight, len(train_clients))
        
        elif args.train_mode == "fedadp":
            flat_tensor, smooth_angle = aggregate_benchmark_fedadp(
                local_model_weight, flatten_model(client_model), list_client, smooth_angle, round)

        else:
            done = 0
            num_cli = len(train_clients)
            # mean_local_losses, std_local_losses = get_mean_losses(train_local_loss, num_cli)
            _, _, std_local_losses = get_mean_losses(
                train_local_loss, num_cli)
            start_loss = [local_inference_loss[i,0] for i in range(num_cli)]
            final_loss = [local_inference_loss[i,1] for i in range(num_cli)]
            start_l, final_l = start_loss.copy(), final_loss.copy()
            if round:
                prev_reward = get_reward(start_loss)
                np_infer_server_loss = np.asarray(start_loss)
                sample = {
                    "reward": prev_reward,
                    "mean_losses": np.mean(start_loss),
                    "std_losses": np.std(start_loss),
                    "max-min": np_infer_server_loss.max() - np_infer_server_loss.min()
                    # "episode_reward": self.episode_reward,
                }
                wandb.log({'dqn_inside/reward': sample})
            dqn_weights = agent.get_action(start_loss, final_loss, std_local_losses, local_n_sample,
                                           dqn_list_epochs, done, clients_id=train_clients, prev_reward=prev_reward)

            # print("Here final output: ", dqn_weights.shape)            
            s_means, s_std, s_epochs, assigned_priorities = standardize_weights(dqn_weights, num_cli)

            flat_tensor = aggregate(local_model_weight, len(train_clients), assigned_priorities)

            # Update epochs
            if args.train_mode == "RL-Hybrid":
                dqn_list_epochs = s_epochs
                load_epoch(list_client, dqn_list_epochs)

        client_model.load_state_dict(unflatten_model(flat_tensor, client_model))
        # >>>> Test model
        acc, test_loss = test(client_model, DataLoader(test_dataset, 32, False))
        local_loss = [0 for _ in range(len(train_clients))]

        # for i in range(len(train_clients)):
        #     test_args = (
        #             i,
        #             train_clients[i],
        #             copy.deepcopy(client_model),
        #             list_client[train_clients[i]],
        #             local_model_weight,
        #             local_loss
        #         )
        #     test_local(test_args)
        # print(local_loss)
        # for i in range(len(train_clients)):
        #     local_loss[i] = local_inference_loss[i, 0]
        # prev_reward = get_reward(local_loss)

        print("ROUND: ", round, " TEST ACC: ", acc)

        train_time, delay, max_time, min_time = get_train_time(local_n_sample, list_abiprocess)

        if args.train_mode in ["benchmark", "fedadp"]:
            logging = {
                "round": round + 1,
                "clients_per_round": args.clients_per_round,    
                "n_epochs": args.num_epochs,
                "local_train_time": max_time,
                "delay": delay,
                "test_loss": test_loss
            }
            wandb.log({'test_acc': acc, 'summary/summary': logging})

        else:
            dictionaryLosses = getDictionaryLosses(start_l, final_l, num_cli)
            logging = {
                "round": round + 1,
                "clients_per_round": args.clients_per_round,
                "n_epochs": args.num_epochs,
                "local_info": dictionaryLosses,
                "local_train_time": max_time,
                "delay": delay,
                "test_loss": test_loss,
            }
            dqn_sample = {
                "means": s_means,
                "std": s_std,
                "num_epochs": s_epochs,
                "assigned_priorities": assigned_priorities,
            }
            recordedSample = getLoggingDictionary(dqn_sample, num_cli)
            wandb.log({'test_acc': acc, 'dqn/dqn_sample': recordedSample, 'summary/summary': logging})

    del pool


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parse_args = option()

    wandb.init(project="federated-learning-dqn",
               entity="aiotlab",
               name=parse_args.run_name,
               group=parse_args.group_name,
               #    mode="disabled",
               config={
                   "num_rounds": parse_args.num_rounds,
                   "num_clients": parse_args.num_clients,
                   "clients_per_round": parse_args.clients_per_round,
                   "batch_size": parse_args.batch_size,
                   "num_epochs": parse_args.num_epochs,
                   "path_data_idx": parse_args.path_data_idx,
                   "learning_rate": parse_args.learning_rate,
                   "algorithm": parse_args.algorithm,
                   "mu": parse_args.mu,
                   "seed": parse_args.seed,
                   "drop_percent": parse_args.drop_percent,
                   "num_core": parse_args.num_core,
                   "log_dir": parse_args.log_dir,
                   "train_mode": parse_args.train_mode,
                   "dataset_name": parse_args.dataset_name,
                   "beta": parse_args.beta,
                    "hidden_dim": parse_args.hidden_dim,
                    "init_w": parse_args.init_w,
                    "value_lr": parse_args.value_lr,
                    "policy_lr": parse_args.policy_lr,
                    "max_steps": parse_args.max_steps,
                    "max_frames": parse_args.max_frames,
                    "batch_size_ddpg": parse_args.batch_size_ddpg,
                    "gamma": parse_args.gamma,
                    "soft_tau": parse_args.soft_tau,
               })
    args = wandb.config
    wandb.define_metric("test_acc", summary="max")
    print(">>> START RUNNING: {} - Train mode: {} - Dataset: {}".format(parse_args.run_name, args.train_mode, args.dataset_name))
    main(args)
    wandb.finish()
