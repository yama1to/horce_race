
from dis import dis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from explorer import common 

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        
        # self.conv1 = nn.Conv2d(1000, 1000, 10, 1)
        # self.conv2 = nn.Conv2d(1000,1000, 10, 1)
        # self.pool = nn.MaxPool2d(10,stride = 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.log = nn.LogSoftmax()
        #self.conv2 = nn.Conv2d(1000,, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(input_size, 5000)
        self.fc2 = nn.Linear(5000, 500)
        self.fc3 = nn.Linear(500, 16)
        self.fc4 = nn.Linear(16, output_size)

        # self.fc11 = nn.Linear(100, 1000)
        # self.fc12 = nn.Linear(1000, 1000)
        # self.fc13 = nn.Linear(1000, 100)


    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        #x=self.sigmoid(x)
        #x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.tanh(x)

        x = self.fc3(x)
        #x=self.tanh(x)
        x = self.relu(x)
        #x=self.sigmoid(x)
        x = self.dropout1(x)

  

        x = self.fc4(x)
        x = self.sigmoid(x)    
        #x = F.log_softmax(x, dim=1)
        #x=self.tanh(x)
        
        #print(x)
        #print(x)
        #x = F.log_softmax(x, dim=1)
                # x = self.fc2(x)
        # x=self.tanh(x)  #x = self.relu(x)
        # x = self.fc2(x)
        # x=self.tanh(x)  #x = self.relu(x)
        # x = self.fc11(x)
        # x=self.tanh(x)
        # x = self.fc12(x)
        # x=self.tanh(x)
        # x = self.fc13(x)
        # x=self.tanh(x)

        #x=self.tanh(x) #x=self.sigmoid(x)
        return x