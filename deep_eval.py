import torch
from dis import dis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from explorer import common 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def prepare_data_is_tansyo():
    target_name='is_tansyo'
    final_df = pd.read_csv("csv/test1.csv", sep=",")


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

model_path = '/home/test/Downloads/sakino0509/race-predict/model/20220527_130535_model_weights.pth'
model = torch.load(model_path)

#model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_inputs,test_labels = prepare_data_is_tansyo()
num = len(test_inputs)
image_size = 16

criterion = nn.MSELoss()

with torch.no_grad():
    inputs, labels = test_inputs[:16], test_labels[:16]

    # GPUが使えるならGPUにデータを送る
    inputs = inputs.to(device)
    labels = labels.to(device)

    # ニューラルネットワークの処理を行う
    inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
    outputs = model(inputs)

    # 損失(出力とラベルとの誤差)の計算
    loss_sum += criterion(outputs, labels)

    # 正解の値を取得
    pred = outputs.argmax(1)
    # 正解数をカウント
    correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum /(num//16)}, Accuracy: {100*correct/(num//16)}% ({correct})")
