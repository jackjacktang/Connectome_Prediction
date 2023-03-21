import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class mlp(nn.Module):
    def __init__(self, in_ch, out_ch, br=True):
        super(mlp, self).__init__()
        self.layer = nn.Linear(in_ch, out_ch)
        self.br = br
        self.bn = nn.BatchNorm1d(out_ch)
        # self.bn = nn.GroupNorm(2, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer(x)
        if self.br:
            x = self.bn(x)
            x = self.relu(x)

        return x


class connect_net(nn.Module):
    def __init__(self):
        super(connect_net, self).__init__()
        # self.encoder1 = encoder_layer(1, 16, 7)
        # self.encoder2 = encoder_layer(16, 64, 5)
        # self.encoder3 = encoder_layer(64, 256, 3)
        self.fc1 = mlp(3486, 1024)
        self.fc2 = mlp(1024, 256)
        self.fc3 = mlp(256, 64)
        self.fc4 = mlp(64, 2, br=False)

    def forward(self, x):
        # x = self.encoder1(x)
        # x = self.encoder2(x)
        # x = self.encoder3(x)
        # print(x.size())
        # x = torch.flatten(x, 1)
        # print('after flatten')
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        x = self.fc3(x)
        # print(x.size())
        x = self.fc4(x)
        # print(x.size())

        return x
