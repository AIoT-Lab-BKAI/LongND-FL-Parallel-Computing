python3 -u train_SGD.py --log_file='fedavg-mnist-non-iid' --algorithm='fedavg' --num_clients=10 --clients_per_round=10 --num_rounds=1000 --learning_rate=0.01 --batch_size=10 --num_epochs=5 --num_samples_per_class=980 --log_file='log/non-iid/fedavg'