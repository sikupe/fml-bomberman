import torch
import torch.nn as nn


class QNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(QNN, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size).double()

    def forward(self, x):
        x = self.layer1(x)
        return torch.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num
