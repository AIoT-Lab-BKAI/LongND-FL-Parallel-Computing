import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
  CNN with two 5x5 convolution lauers(the first with 32 channels, second with 64,
  each followed with 2x2 max pooling), a fully connected layer with 512 uunits and 
  ReLu activation, and the final Softmax output layer

  Total Expected Params: 1,663,370
  """

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        out = F.log_softmax(x, dim=1)
        return out


class MNIST_2NN(nn.Module):
    """
  A simple multilayer-perceptron with 2-hidden layers with 200 units each
  using ReLu activations
  Total Expected Params: 199,210
  """

    def __init__(self):
        super(MNIST_2NN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class CNNCifarRestnet50(nn.Module):
    def __init__(self, pretrained=False, trainable=True, num_classes=100):
        super(CNNCifarRestnet50, self).__init__()
        model_conv = torchvision.models.resnet50(pretrained=pretrained)
        for param in model_conv.parameters():
            param.requires_grad = trainable
        model_conv.fc = nn.Linear(model_conv.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.model_conv(x)
        return F.log_softmax(x, dim=1)

