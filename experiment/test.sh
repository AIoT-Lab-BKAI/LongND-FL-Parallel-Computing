python3 -u train.py --num_clients=10 --clients_per_round=10 --num_rounds=50 --learning_rate=0.01 --batch_size=16 --num_epochs=1 --train_mode="fedadp" --dataset_name="mnist" --path_data_idx="dataset_idx/mnist/equal/MNIST-noniid-fedavg_equal_1.json" --run_name="FedAdp-FedAVG-Equal-01" --group_name="Test" --num_core=1
