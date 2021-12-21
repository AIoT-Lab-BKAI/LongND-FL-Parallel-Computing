# LongND-FL-Parallel-Computing

## Data
* Dữ liệu train và test được download tự động từ torchvision datasets.
* Thực nghiệm trên dữ liệu Mnist

## Run
```
python python train_SGD.py
```

## Options
Giá trị tham số được lấy trong options.py. Chi tiết một số tham số:
* ```--num_rounds``` Default: 1. Số round của huấn luyện.
* ```--num_clients:``` Default: 1000. Tổng số client.
* ```--clients_per_round``` Default: 4. Số client được chọn để tham gia train mỗi round.
* ```--drop_percent``` Default: 0.1. Tỷ lệ client được chọn không thể tham gia train mỗi round.
* ```num_samples_per_client``` Default: 10. Số sample dữ liệu của mỗi client.
* ```num_class_per_client``` Default: 2. Số class mà mỗi client có.
* ```rate_balance``` Default:0. Tỷ lệ không cân bằng dữ liệu. Số dữ mỗi client có là ((1+(num_class_per_client-1)*rate_balance))*num_samples_per_client
* ```--path_data_idx``` Địa chỉ lưu idx dữ liệu của từng client.
* ```--load_data_idx``` Default: False. Có tải dữ liệu idx của client đã có sẵn hay không.
* ```--mu``` Default:0. Giá trị trọng số µ trong Fedprox loss. FedAvg là trường hợp đặc biệt của FedProx khi µ = 0.

## File 
* ```dataset_idx.json``` Lưu thông tin idx dữ liệu của từng client.
* ```abi_process``` Lưu thông tin khả năng tính toán của từng client.
* ```round.txt``` Lưu những client được train mỗi round.
* ```log.json``` Lưu những thông tin chi tiết trong lúc train.
