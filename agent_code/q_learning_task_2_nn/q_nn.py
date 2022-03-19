import torch
import torch.nn as nn
import torch.nn.functional as F


class QNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(QNN, self).__init__()
        intermediate_size = int(input_size * 1.5)

        self.layer1 = nn.Linear(input_size, intermediate_size).double()
        self.layer2 = nn.Linear(intermediate_size, output_size).double()

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num
