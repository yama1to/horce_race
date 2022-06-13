import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from explorer import common 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

from sklearn import utils
from NN import Net

from tqdm import tqdm 

# path = 'model/20220528_230441_model_1.pth'
# path = './model/20220529_002554_model_weights.pth'
path = '/home/test/Downloads/sakino0509/race-predict/model/20220529_050223/20220529_050223_model_epoch=2::Accu=0.09.pth'
image_size = 38     # 画像の画素数(幅x高さ)
output_size = 1

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data = "csv/data3.csv"
data = 'csv/data_race_tokyo.csv'


criterion = nn.MSELoss()

input = pd.read_csv(data)
print(input.columns,len(input.columns))
input = input.drop(['race_id',], axis=1).values
sc = StandardScaler()
input = sc.fit_transform(input)
input = torch.tensor(input)

loss_sum = 0
correct = 0
num = len(input)
ep = num//18


model = Net(image_size, output_size)
model.load_state_dict(torch.load(path))


import numpy as np
with torch.no_grad():
    for i in tqdm(range(ep)):
        inputs= input[i:i+18]
        #inputs, labels = utils.shuffle(inputs, labels)
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        #labels = labels.to(device)
        #print(inputs.shape)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)
        print(outputs)
        #print(outputs.shape)
        # 損失(出力とラベルとの誤差)の計算
        #print(outputs)
        pred = outputs.argmax()
        zero = torch.zeros(18)

        #loss_sum += criterion(outputs, labels)
        #print(outputs)
        
        # 正解の値を取得
        argmax = outputs.argmax()
        zeros = torch.zeros((18,1))
        zeros[argmax] = 1
        #print(outputs)
        pred = zeros
        print(argmax+1)
        #pred = outputs.argmax(1)
        
        # print(np.argmax(outputs[0]))
        # 正解数をカウント
        #eq = pred.eq(labels.view_as(pred)).sum().item()
        # print(pred)
        # print(labels.view_as(pred))
        #correct += eq//18
        # print(labels.view_as(pred))
        # print(pred)
        #print(correct)
        #if correct==1:
            #print('正解:'+str(np.argmax(labels.numpy()))+'予測:'+str(np.argmax(pred.view_as(pred).numpy())))
