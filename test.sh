CUDA_VISIBLE_DEVICES=0 python train.py --train_mode="benchmark" --dataset_name="mnist" --num_clients=10 --clients_per_round=10 --num_rounds=2 --learning_rate=0.01 --batch_size=10 --num_epochs=1 --path_data_idx="dataset_idx/mnist/equal/MNIST-noniid-fedavg_equal_2.json" --run_name="FedAVG-Equal-Non-IID-02" --group_name="MNIST-FedAvg-Equal" --num_core 1