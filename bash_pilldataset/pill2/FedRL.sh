python train.py --train_mode="RL-Fixed" --dataset_name="pill_dataset" --num_clients=10 --clients_per_round=10 --num_rounds=1000 --learning_rate=0.01 --batch_size=5 --num_epochs=5 --run_name="FedRL-Pill-Dataset" --group_name="Pill-Dataset2-ICPP" --num_core 1 --pill_dataset_path "pill2/" --pill_datasetidx "syndata2"