import argparse


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_rounds", help="number of rounds to simulate;", type=int, default=1000
    )
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument(
        "--clients_per_round",
        help="number of clients trained per round;",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size when clients train on data;",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_epochs",
        help="number of epochs when clients train on data;",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate for inner solver;",
        type=float,
        default=0.003,
    )
    parser.add_argument("--mu", help="constant for prox;",
                        type=float, default=0.01)
    parser.add_argument("--algorithm", type=str, default="")
    parser.add_argument(
        "--seed", help="seed for randomness;", type=int, default=10)
    parser.add_argument(
        "--drop_percent", help="percentage of slow devices", type=float, default=0.0
    )
    parser.add_argument("--num_core", type=int, default=9)
    parser.add_argument("--log_dir", type=str, default='./')
    parser.add_argument("--dataset_name", type=str, default="mnist")
    parser.add_argument("--path_data_idx", type=str,
                        default="dataset_idx/mnist/equal/MNIST-noniid-fedavg_equal_1.json")
    parser.add_argument("--run_name", type=str, default="dung-debug")
    parser.add_argument("--group_name", type=str, default="dung")
    parser.add_argument("--train_mode", type=str, default="RL-Hybrid")

    args = parser.parse_args()
    return args
