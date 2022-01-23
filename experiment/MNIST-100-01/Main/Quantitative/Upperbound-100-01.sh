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

python3 -u train_upperbound.py --train_mode="benchmark" --dataset_name="mnist" --num_rounds=1000 --learning_rate=0.01 --batch_size=10 --path_data_idx="dataset_idx/mnist/100client/quantitative/MNIST-noniid-quantitative_1_all_data.json" --run_name="Upperbound-Quantitative-100-01" --group_name="MNIST-100-10"
