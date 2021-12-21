import argparse
from os import stat
from numpy.core.arrayprint import str_format
from numpy.core.defchararray import count
from numpy.lib.function_base import _percentile_dispatcher
from numpy.lib.npyio import save
from torchvision.datasets import mnist
from torchvision import transforms, datasets
from modules.Client import Client
from utils.utils import (
    GenerateLocalEpochs,
    save_infor,
    select_client,
    select_drop_client,
)
from utils.loader import iid_partition, non_iid_partition, mnist_extr_noniid
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


def main():
    """ Parse command line arguments or load defaults """
    args = option()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    logname = "round.txt"
    logging.basicConfig(
        filename=logname,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    generate_abiprocess(mu=100, sigma=5, n_client=args.num_clients)
    list_abiprocess_client = read_abiprocesss()
    assert len(list_abiprocess_client) == args.num_clients, "list abiprocesss k du"

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
    if args.load_data_idx:
        list_idx_sample = load_dataset_idx(args.path_data_idx)
    else:
        list_idx_sample = mnist_extr_noniid(train_dataset, args.num_clients,args.num_class_per_client,args.num_samples_per_client,args.rate_balance)
        save_dataset_idx(list_idx_sample, args.path_data_idx)

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
    list_abiprocess = []
    list_sam = []

    for round in range(args.num_rounds):
        print("Train :------------------------------")
        # Ngau nhien lua chon client de train
        selected_client = select_client(args.num_clients, args.clients_per_round)
        drop_clients, train_client = select_drop_client(
            selected_client, args.drop_percent
        )
        train_clients = list(set(selected_client) - set(drop_clients))
        # Khoi tao cac bien su dung train
        local_model_weight = torch.zeros(len(train_clients), n_params)
        local_model_weight.share_memory_()

        train_local_loss = torch.zeros(len(train_client), args.num_epochs)
        train_local_loss.share_memory_()
        list_trained_client.append(train_clients)
        list_abiprocess.append([list_client[i].abiprocess for i in train_clients])
        local_n_sample = [list_client[i].n_samples for i in train_clients]
        str_sltc = ""
        for i in train_clients:
            str_sltc += str(i) + " "
        logging.info(f"Round {round} Selected client : {str_sltc} ")

        # Huan luyen song song tren cac client
        with mp.Pool(args.num_core) as pool:
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
        flat_tensor = aggregate(local_model_weight, len(train_clients))
        mnist_cnn.load_state_dict(unflatten_model(flat_tensor, mnist_cnn))
        # Testcccccccccccccccccc
        acc = test(mnist_cnn, DataLoader(test_dataset, 32, False))
        train_time, delay, max_time, min_time = get_train_time(
            local_n_sample, list_abiprocess
        )
        sample = {
            "round": round + 1,
            "clients_per_round": args.clients_per_round,
            "n_epochs": args.num_epochs,
            "selected_clients": list([int(i) for i in selected_client]),
            "drop_clients": list([int(i) for i in drop_clients]),
            "local_loss": convert_tensor_to_list(train_local_loss),
            "local_train_time": max_time,
            "delay": delay,
            "accuracy": acc,
        }
        list_sam.append(sample)
    save_infor(list_sam, "log.json")


if __name__ == "__main__":
    main()

