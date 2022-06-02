from torchvision import transforms, datasets

apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10("./data/cifar/", train=True, download=True, transform=apply_transform)
test_dataset = datasets.CIFAR10("./data/cifar/", train=False, download=True, transform=apply_transform)

