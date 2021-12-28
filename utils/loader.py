from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


def iid_partition(dataset, clients):
    """
  I.I.D paritioning of data over clients
  Shuffle the data
  Split it between clients

  params:
    - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
    - clients (int): Number of Clients to split the data between

  returns:
    - Dictionary of image indexes for each client
  """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        tmp = set(np.random.choice(
            image_idxs, num_items_per_client, replace=False))
        client_dict[i] = [int(i) for i in tmp]

        image_idxs = list(set(image_idxs) - tmp)

    return client_dict


def non_iid_partition(
    dataset, clients, total_shards, shards_size, num_shards_per_client
):
    """
  non I.I.D parititioning of data over clients
  Sort the data by the digit label
  Divide the data into N shards of size S
  Each of the clients will get X shards

  params:
    - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
    - clients (int): Number of Clients to split the data between
    - total_shards (int): Number of shards to partition the data in
    - shards_size (int): Size of each shard 
    - num_shards_per_client (int): Number of shards of size shards_size that each client receives

  returns:
    - Dictionary of image indexes for each client
  """

    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype="int64") for i in range(clients)}
    idxs = np.arange(len(dataset))
    data_labels = dataset.targets.numpy()

    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(clients):
        rand_set = set(
            np.random.choice(shard_idxs, num_shards_per_client, replace=False)
        )
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate(
                (client_dict[i], idxs[rand *
                 shards_size: (rand + 1) * shards_size]),
                axis=0,
            )
    return client_dict


def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = "../data/mnist/"
    apply_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=apply_transform
    )

    # Chose euqal splits for every user
    user_groups_train = mnist_extr_noniid(
        train_dataset, num_users, n_class, nsamples, rate_unbalance
    )
    return train_dataset, user_groups_train


def mnist_extr_noniid(train_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(60000 / num_samples), num_samples
    num_classes = 10
    assert n_class * num_users <= num_shards_train
    assert n_class <= num_classes
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[rand *
                             num_imgs_train: (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[rand *
                               num_imgs_train: (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
            else:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[
                            rand
                            * num_imgs_train: int(
                                (rand + rate_unbalance) * num_imgs_train
                            )
                        ],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[
                            rand
                            * num_imgs_train: int(
                                (rand + rate_unbalance) * num_imgs_train
                            )
                        ],
                    ),
                    axis=0,
                )
            unbalance_flag = 1
            idxi = dict_users_train[i]
            dict_users_train[i] = [int(i) for i in idxi]
    return dict_users_train


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
