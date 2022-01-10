#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=36:00:00
#$-j y
#$-cwd
source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.0/11.0.3 cudnn/8.0/8.0.5
source /home/acc13085dy/federated-learning/FLenv/bin/activate
cd /home/acc13085dy/federated-learning/LongND-FL-Parallel-Computing

python3 -u train_benchmark.py --num_clients=10 --clients_per_round=10 --num_rounds=2000 --learning_rate=0.01 --batch_size=10 --num_epochs=5 --path_data_idx="dataset_idx/MNIST-featured_non-iid.txt" --run_name="Benchmark-FedProx-MNIST-Featured" --group_name="Benchmark" --algorithm="fedprox" --mu=0.01

