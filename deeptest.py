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
path = '/home/test/Downloads/sakino0509/race-predict/model/20220529_004750_model_weights.pth'

image_size = 48-4      # 画像の画素数(幅x高さ)
output_size = 1

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data = "csv/data3.csv"
data = 'csv/data_race_tokyo.csv'


def prepare_data_is_tansyo():
    target_name='is_tansyo'
    final_df = pd.read_csv(data, sep=",")
    final_df = final_df[:,1:]
    #print(final_df.shape)
    # print(final_df.columns)
    # exit()


    train_ratio = 0.8
    X =final_df.sort_values("date")
    train_size = int(len(X) * train_ratio)
    train_df = X[0:train_size].copy().reset_index(drop=True)
    test_df = X[train_size:len(X)].copy().reset_index(drop=True)

    Y_train = train_df[target_name].values
    X_train = train_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    Y_test = test_df[target_name].values
    X_test = test_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)



  
    X_train, Y_train, X_test, Y_test = torch.tensor(X_train),torch.tensor(Y_train),torch.tensor(X_test),torch.tensor(Y_test)
    return X_train, Y_train, X_test, Y_test


_,_,test_inputs, test_labels = prepare_data_is_tansyo()
criterion = nn.MSELoss()

loss_sum = 0
correct = 0
num = len(test_inputs)
ep = num//16


model = Net(image_size, output_size)
model.load_state_dict(torch.load(path))
import numpy as np
with torch.no_grad():
    for i in tqdm(range(ep)):
        inputs, labels = test_inputs[i:i+16], test_labels[i:i+16]
        #inputs, labels = utils.shuffle(inputs, labels)
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)
        #print(outputs.shape)
        # 損失(出力とラベルとの誤差)の計算
        pred = outputs.argmax(0)
        zero = torch.zeros(16)

        loss_sum += criterion(outputs, labels)
        #print(outputs)
        
        # 正解の値を取得
        argmax = outputs.argmax()
        zeros = torch.zeros((16,1))
        zeros[argmax] = 1
        #print(outputs)
        pred = zeros

        #pred = outputs.argmax(1)
        
        # print(np.argmax(outputs[0]))
        # 正解数をカウント
        eq = pred.eq(labels.view_as(pred)).sum().item()
        # print(pred)
        # print(labels.view_as(pred))
        correct += eq//16
        # print(labels.view_as(pred))
        # print(pred)
        #print(correct)
        if correct==1:
            print('正解:'+str(np.argmax(labels.numpy()))+'予測:'+str(np.argmax(pred.view_as(pred).numpy())))


print(f"Loss: {loss_sum /(ep)}, Accuracy: {100*correct/(ep)}% ({correct})")
