import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        # self.device = device
        
    def forward(self, state, action):
        # print(f'state: {state.shape}')
        # print(f'ACTION: {action.shape}')
        # if ()
        # state = state.squeeze()
        x = torch.cat([state, action], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-3):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.tanh = nn.Tanh()
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        # print('hihi')
        x = F.relu(self.linear1(state))
        # print(x)
        x = F.relu(self.linear2(x))
        # print(x)
        # x = F.tanh(self.linear3(x))
        x = self.tanh(self.linear3(x))
        # print(x)
        return x
    
    def get_action(self, state):
        # device = torch.device("cuda" if use_cuda else "cpu")
        # state = torch.DoubleTensor(state).unsqueeze(0)
        # print(f'state: {state}')
        action = self.forward(state)
        # print(action)
        return action.detach().cpu().numpy()

if __name__ == '__main__':
    model = PolicyNetwork(6,6,32)
    state = torch.tensor([[  100000.0,   -100000.0, -100000.0,   -100000.0,   100000.0,  100000.0]])
    action = model.get_action(state)
    print(action)
