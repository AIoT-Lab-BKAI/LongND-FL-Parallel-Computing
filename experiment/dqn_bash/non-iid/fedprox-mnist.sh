CUDA_VISIBLE_DEVICES=1 python3 -u train_SGD.py --logs_file='fedprox-mnist' --algorithm='fedprox' --num_clients=10 --clients_per_round=10 --num_rounds=100 --learning_rate=0.01 --batch_size=10 --num_epochs=5 --num_samples_per_class=980 --mu=0.1 --log_file='log/epochs'