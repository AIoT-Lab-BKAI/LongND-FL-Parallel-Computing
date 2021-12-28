import argparse


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_rounds", help="number of rounds to simulate;", type=int, default=2
    )
    parser.add_argument(
        "--eval_every", help="evaluate every ____ rounds;", type=int, default=-1
    )
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument(
        "--clients_per_round",
        help="number of clients trained per round;",
        type=int,
        default=4,
    )
    # parser.add_argument("--", type=int, default=10)
    parser.add_argument("--num_class_per_client", type=int, default=2)
    parser.add_argument("--rate_balance", type=int,
                        default=0, help="0 is unbalance")
    parser.add_argument(
        "--batch_size",
        help="batch size when clients train on data;",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_epochs",
        help="number of epochs when clients train on data;",
        type=int,
        default=10,
    )
    parser.add_argument("--path_data_idx", type=str,
                        default="dataset_idx.json")
    parser.add_argument("--load_data_idx", type=bool, default=False)
    parser.add_argument(
        "--learning_rate",
        help="learning rate for inner solver;",
        type=float,
        default=0.003,
    )
    parser.add_argument("--num_samples_per_client", type=int, default=10)
    parser.add_argument("--mu", help="constant for prox;",
                        type=float, default=0.1)
    parser.add_argument(
        "--seed", help="seed for randomness;", type=int, default=10)
    parser.add_argument(
        "--drop_percent", help="percentage of slow devices", type=float, default=0.1
    )
    parser.add_argument("--algorithm", type=str, default="fedprox")
    parser.add_argument("--num_core", type=int, default=2)

    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--logs_file", type=str, default="logs")
    args = parser.parse_args()
    return args
