import argparse
from math import ceil
import os.path
from os import path
from os import stat
from numpy.core.arrayprint import str_format
from numpy.core.defchararray import count
from numpy.lib.function_base import _percentile_dispatcher
from numpy.lib.npyio import save
from torchvision.datasets import mnist
from torchvision import transforms, datasets
from modules.Client import Client
from tqdm import tqdm
from utils.utils import (
    GenerateLocalEpochs,
    get_mean_losses,
    getDictionaryLosses,
    getLoggingDictionary,
    save_infor,
    select_client,
    select_drop_client,
    standardize_weights,
)
from utils.loader import iid_partition, non_iid_partition, mnist_extr_noniid, mnist_noniid_client_level
from utils.trainer import train
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import copy
import logging
from utils.trainer import test
from torch.utils.data import DataLoader
from utils.utils import (
    read_abiprocesss,
    generate_abiprocess,
    save_dataset_idx,
    load_dataset_idx,
    convert_tensor_to_list,
    get_train_time,
    count_params,
    aggregate,
    unflatten_model,
    flatten_model,
)
from utils.option import option
from models.models import MNIST_CNN
from ddpg_agent.ddpg import *

from utils.utils import load_epoch, log_by_round
import wandb


def main(args):
    """ Parse command line arguments or load defaults """
    args = option()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    if args.local_save_mode:
        logname = args.log_dir + "/" + args.log_file + "_round.txt"
        logging.basicConfig(
            filename=logname,
            filemode="a",
            format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
        path_to_save_log = './'
        if args.log_dir:
            print(args.log_dir)
            if not path.exists(args.log_dir):
                os.mkdir(args.log_dir, mode=0o777, dir_fd=None)
            path_to_save_log = args.log_dir

    generate_abiprocess(mu=100, sigma=5, n_client=args.num_clients)
    list_abiprocess_client = read_abiprocesss()
    assert len(
        list_abiprocess_client) == args.num_clients, "not enough abi-processes"

    transforms_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "./data/mnist/", train=True, download=True, transform=transforms_mnist
    )
    test_dataset = datasets.MNIST(
        "../data/mnist/", train=False, download=True, transform=transforms_mnist
    )
    mnist_cnn = MNIST_CNN()

    list_idx_sample = load_dataset_idx(args.path_data_idx)

    # exit()
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
    n_params = count_params(mnist_cnn)
    list_trained_client = []

    list_sam = []

    # Agent to get next settings for this round
    # This is dimensions' configurations for the DQN agent
    state_dim = args.num_clients * 3  # each agent {L, e, n}
    # action_dim = args.num_clients * 2
    # plus action for numbers of epochs for each client
    action_dim = args.num_clients * 3

    agent = DDPG_Agent(state_dim=state_dim,
                       action_dim=action_dim, log_dir=args.log_dir).cuda()

    # TODO: Khởi tạo multi-process
    pool = mp.Pool(args.num_core)

    for round in tqdm(range(args.num_rounds)):
        # mocking the number of epochs that are assigned for each client.
        dqn_list_epochs = [args.num_epochs for _ in range(args.num_clients)]
        # Ngau nhien lua chon client de train
        selected_client = select_client(
            args.num_clients, args.clients_per_round)
        drop_clients, train_client = select_drop_client(
            selected_client, args.drop_percent
        )
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
        str_sltc = ""
        for i in train_clients:
            str_sltc += str(i) + " "

        if args.local_save_mode:
            logging.info(f"Round {round} Selected client : {str_sltc} ")

        # Huan luyen song song tren cac client
        pool.map(
            train,
            [
                (
                    i,
                    train_clients[i],
                    copy.deepcopy(mnist_cnn),
                    list_client[train_clients[i]],
                    local_model_weight,
                    train_local_loss,
                    args.algorithm,
                )
                for i in range(len(train_clients))
            ],
        )

        # FedAvg weight local model va cap nhat weight global
        done = 0
        num_cli = len(train_clients)
        mean_local_losses = get_mean_losses(train_local_loss, num_cli)
        dqn_weights = agent.get_action(
            mean_local_losses, local_n_sample, dqn_list_epochs, done)
        s_means, s_std, s_epochs, assigned_priorities = standardize_weights(
            dqn_weights, num_cli)

        # Update Epochs
        # dqn_list_epochs = s_epochs
        flat_tensor = aggregate(local_model_weight, len(
            train_clients), assigned_priorities)
        mnist_cnn.load_state_dict(unflatten_model(flat_tensor, mnist_cnn))

        # Test on test set
        acc, test_loss = test(mnist_cnn, DataLoader(test_dataset, 32, False))

        train_time, delay, max_time, min_time = get_train_time(
            local_n_sample, list_abiprocess
        )
        # logging_dqn_weights = get_info_from_dqn_weights(dqn_weights, len(train_clients), dqn_list_epochs)

        dictionaryLosses = getDictionaryLosses(np.asarray(
            mean_local_losses).reshape((num_cli)), num_cli)

        sample = {
            "round": round + 1,
            "clients_per_round": args.clients_per_round,
            "n_epochs": args.num_epochs,
            "local_train_loss": dictionaryLosses,
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
        list_sam.append(sample)
        if args.local_save_mode:
            log_by_round(sample, path_to_save_log+"/round_log.json")

        # load_epoch(list_client, dqn_list_epochs)

        wandb.log({'test_acc': acc, 'dqn/dqn_sample': recordedSample,
                  'summary/summary': sample})

    if args.local_save_mode:
        save_infor(list_sam, path_to_save_log+"/log.json")

    del pool


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parse_args = option()

    wandb.init(project="dungnt-federated-learning-dqn",
               entity="aiotlab",
               name=parse_args.run_name,
               group=parse_args.group_name,
               #    mode="disabled",
               config={
                   "num_rounds": parse_args.num_rounds,
                   "eval_every": parse_args.eval_every,
                   "num_clients": parse_args.num_clients,
                   "clients_per_round": parse_args.clients_per_round,
                   "num_class_per_client": parse_args.num_class_per_client,
                   "rate_balance": parse_args.rate_balance,
                   "batch_size": parse_args.batch_size,
                   "num_epochs": parse_args.num_epochs,
                   "path_data_idx": parse_args.path_data_idx,
                   "load_data_idx": parse_args.load_data_idx,
                   "learning_rate": parse_args.learning_rate,
                   "num_samples_per_class": parse_args.num_samples_per_class,
                   "mu": parse_args.mu,
                   "seed": parse_args.seed,
                   "drop_percent": parse_args.drop_percent,
                   "algorithm": parse_args.algorithm,
                   "num_core": parse_args.num_core,
                   "log_dir": parse_args.log_dir,
                   "log_file": parse_args.log_file,
                   "num_sample_per_class": parse_args.num_samples_per_class,
                   "local_save_mode": parse_args.local_save_mode
               })

    args = wandb.config
    wandb.define_metric("test_acc", summary="max")

    main(args)
    wandb.finish()
