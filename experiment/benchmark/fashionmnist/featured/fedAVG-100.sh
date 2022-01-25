#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=36:00:00
#$-j y
#$-cwd
source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source /home/acc13085dy/federated-learning/FLenv/bin/activate

cp -rp /home/acc13085dy/federated-learning/LongND-FL-Parallel-Computing $SGE_LOCALDIR/$JOB_ID/
cd $SGE_LOCALDIR/$JOB_ID

python3 -u train.py --train_mode="benchmark" --dataset_name="fashionmnist" --num_clients=100 --clients_per_round=10 --num_rounds=1000 --learning_rate=0.01 --batch_size=10 --num_epochs=5 --path_data_idx="dataset_idx/fashionmnist/100client/featured/FashionMNIST-noniid-featured_1.json" --run_name="fedAVG-Featured-100-Clients" --group_name="FASHION-MNIST-Featured"
