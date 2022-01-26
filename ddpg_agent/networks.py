import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-1):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, batch_size):

        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-1):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.tanh = nn.Tanh()

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        

    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x

    def get_action(self, state):
        action = self.forward(state)
        return action.detach().cpu()


if __name__ == '__main__':
    model = ValueNetwork(10 * 4, 10 * 3, 256)

    state = torch.ones(size=[4, 40])
    action = torch.ones(size=[12, 10])

    value = model(state, action, 4)
    print(value.shape)
