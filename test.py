from torchvision import transforms, datasets

transforms_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
train_dataset = datasets.MNIST(
        "./data/mnist/", train=True, download=True, transform=transforms_mnist
    )
test_dataset = datasets.MNIST(
        "../data/mnist/", train=False, download=True, transform=transforms_mnist
    )

list_label = train_dataset.targets.numpy()
# for i in train_dataset:
#     print(i[1])
from utils.loader import mnist_noniid_client_level
from utils.utils import save_dataset_idx
idx = mnist_noniid_client_level(train_dataset,5)
# for i in/
save_dataset_idx(idx, "test.log")
