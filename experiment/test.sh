python3 -u train.py --num_clients=10 --clients_per_round=10 --num_rounds=200 --learning_rate=0.01 --batch_size=16 --num_epochs=1 --train_mode="FedRL-fixed" --dataset_name="mnist" --path_data_idx="dataset_idx/mnist/100client/equal/MNIST-noniid-fedavg_equal_100.json" --run_name="FedRL-equal-01" --group_name="Upgrade-100-3.0" --num_core=1 --beta=0.5

