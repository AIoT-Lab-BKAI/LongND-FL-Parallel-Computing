python3 -u train_benchmark.py --num_clients=10 --clients_per_round=10 --num_rounds=2000 --learning_rate=0.01 --batch_size=10 --num_epochs=5 --path_data_idx="dataset_idx/MNIST-quantitative_non-iid.txt" --run_name="FedAVG-MNIST-Quantitative" --group_name="Benchmark" --algorithm="fedavg"

