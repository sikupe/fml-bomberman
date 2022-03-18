import torch
import torch.nn as nn
import torch.nn.functional as F


class QNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(QNN, self).__init__()
        intermediate_size = input_size * 2

        self.layer1 = nn.Linear(input_size, intermediate_size).double()
        self.layer2 = nn.Linear(intermediate_size, intermediate_size).double()
        self.layer3 = nn.Linear(intermediate_size, output_size).double()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num
