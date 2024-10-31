import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

def optimize_routes(locations):
    input_dim = locations.shape[1]
    output_dim = len(locations)
    q_network = QNetwork(input_dim, output_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=0.01)
    return q_network, optimizer