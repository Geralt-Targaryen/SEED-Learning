import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, n0=16, n1=16, n2=3, dropout=0.1):
        super(Model, self).__init__()

        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.linear1 = nn.Linear(n0, n1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n1, n2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear2(x))
        return x

    def extract_parameter(self):
        params = []
        params.append(self.linear1.weight.reshape(-1, 1).cpu().detach().numpy())
        params.append(self.linear1.bias.reshape(-1, 1).cpu().detach().numpy())
        params.append(self.linear2.weight.reshape(-1, 1).cpu().detach().numpy())
        params.append(self.linear2.bias.reshape(-1, 1).cpu().detach().numpy())
        return np.concatenate(params).reshape((1, -1))

    def inject_parameter(self, params):
        params = torch.Tensor(params)
        self.linear1.weight = nn.parameter.Parameter(params[:self.n0 * self.n1].reshape(self.n1, self.n0))
        self.linear1.bias = nn.parameter.Parameter(params[self.n0 * self.n1:(self.n0 + 1) * self.n1])
        self.linear2.weight = nn.parameter.Parameter(
            params[(self.n0 + 1) * self.n1:(self.n0 + 1 + self.n2) * self.n1].reshape(self.n2, self.n1))
        self.linear2.bias = nn.parameter.Parameter(params[(self.n0 + 1 + self.n2) * self.n1:(self.n0 + 1 + self.n2) * self.n1 + self.n2])
