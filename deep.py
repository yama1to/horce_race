import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 1)
        self.conv2 = nn.Conv2d(6, 16, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5, 40)  # 5*5 from image dimension
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x


net = Net()
print(net)
random_data = torch.rand((1, 1, 1))
x = net.forward(random_data)

print(x)