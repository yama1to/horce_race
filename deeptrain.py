from cProfile import label
from dis import dis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from explorer import common 
import os 

from sklearn import utils
# (310405, 280)(24555, 283)
#----------------------------------------------------------
# ハイパーパラメータなどの設定値

div_num = 1
num_epochs = 100*div_num        # 学習を繰り返す回数
num_batch = 18         # 一度に処理する画像の枚数
learning_rate = 0.00001   # 学習率
image_size = 38      # 画像の画素数(幅x高さ)
output_size = 1

interval_recode = 1            

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
data = "csv/data4.csv"

import pandas as pd 
from sklearn.preprocessing import StandardScaler

#----------------------------------------------------------
# ニューラルネットワークの生成
from NN import Net

model = Net(image_size, output_size).to(device)
print(model)

#----------------------------------------------------------
# 損失関数の設定
# criterion = nn.CrossEntropyLoss() 
criterion = nn.MSELoss()
#----------------------------------------------------------
# 最適化手法の設定
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする



from tqdm import tqdm 



def prepare_data_is_tansyo():
    target_name='is_tansyo'
    final_df = pd.read_csv(data, sep=",")
    print(final_df.shape)


    train_ratio = 0.8
    # X =final_df.sort_values("date")
    train_size = int(len(final_df) * train_ratio)
    train_df = final_df[0:train_size].copy().reset_index(drop=True)
    test_df = final_df[train_size:len(final_df)].copy().reset_index(drop=True)

    Y_train = train_df[target_name].values
    X_train = train_df.drop(['is_tansyo','is_hukusyo','is_owner_same', 'is_rider_same','is_tamer_same',], axis=1).values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    Y_test = test_df[target_name].values
    X_test = test_df.drop(['is_tansyo','is_hukusyo','is_owner_same', 'is_rider_same','is_tamer_same', ], axis=1).values
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)



  
    X_train, Y_train, X_test, Y_test = torch.tensor(X_train),torch.tensor(Y_train),torch.tensor(X_test),torch.tensor(Y_test)
    return X_train, Y_train, X_test, Y_test

train_inputs, train_labels,test_inputs, test_labels = prepare_data_is_tansyo()
num = len(train_inputs)
model_name = './model/%s_model_weights.pth' % common.string_now()
import numpy as np

def eval():
    model.eval()  # モデルを評価モードにする

    loss_sum1 = 0
    correct = 0
    num = len(test_inputs)
    ep = num//18

    ze_in = torch.zeros((2,image_size))
    ze_la = torch.zeros((2))

    with torch.no_grad():
        for i in tqdm(range(ep)):

            inputs, labels = test_inputs[i:i+16], test_labels[i:i+16]
            #print(inputs.shape,ze_in.shape)
            #print(labels.shape,ze_la.shape)
            inputs = torch.vstack([inputs,ze_in])
            labels = torch.hstack([labels,ze_la])
                
            # GPUが使えるならGPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)

            # ニューラルネットワークの処理を行う
            inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
            outputs = model(inputs)
            

            # 損失(出力とラベルとの誤差)の計算
            loss_sum1 += criterion(outputs, labels)

            # 正解の値を取得

            # 正解数をカウント
            argmax = outputs.argmax()
            #print(argmax)
            labels = labels.cpu()
            zeros = torch.zeros((18,1))
            zeros[argmax] = 1
            #print(outputs)
            pred = zeros
            #print(type(pred),type(labels))
            
            eq = pred.eq(labels.view_as(pred)).sum().item()


            correct += eq//18
    acc = correct/(ep)
    loss1 =loss_sum1 /(ep)
    print(f"Loss: {loss1}, Accuracy: {acc}% ({correct})")
    #print('保存したモデル:'+model_name)
    return acc,loss1


now =common.string_now()
os.mkdir('model/%s' % now)
acclist =[]
train_losslist = []
test_losslist = []
loss_sum = torch.zeros(1)
ep = num//18//div_num

for epoch in tqdm(range(num_epochs),disable=False): # 学習を繰り返し行う
    
    
    acc,loss1 = eval()
    model.train()
    loss_sum = 0
    
    ze_in = torch.zeros((2,image_size))
    ze_la = torch.zeros((2))
    for i in tqdm(range(ep),disable=0):
        inputs, labels = train_inputs[i:i+16], train_labels[i:i+16]

        inputs = torch.vstack([inputs,ze_in])
        labels = torch.hstack([labels,ze_la])
        #print(inputs[:2],labels[:2]
        inputs, labels = utils.shuffle(inputs, labels)
        



        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)
        #print(outputs)
        #print(outputs.shape)
        
        # 損失(出力とラベルとの誤差)の計算
        # print(type(outputs[:,0]),type(torch.tensor(labels[0])))
        #print(outputs[:,0],torch.tensor(labels))
        #print(outputs.shape,labels.shape)

        #loss = criterion(outputs[:,0].to(torch.float32),torch.tensor(labels).to(torch.float32))
        loss = criterion(outputs, labels)
        #loss = loss.to(torch.float32)
        loss_sum += loss
        
        # 勾配の計算
        loss.backward()

        # 重みの更新
        optimizer.step()
        #loss = 0

    if epoch%interval_recode==0:
        
        acclist.append(acc)
        test_losslist.append(loss1.cpu())
        train_losslist.append((loss_sum/ep).cpu().detach().numpy())
        model_name = './model/%s/%s_model_epoch=%d::Accu=%.2lf.pth' % (now,now,epoch,acc)

    # 学習状況の表示

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum / (ep)}")

    # モデルの重みの保存
    #torch.save(model.state_dict(),'./model/model_weights.pth')
    torch.save(model.state_dict(), model_name)

import matplotlib.pyplot as plt 

DIR = 'savefigure/'
# print(range(0,epoch,interval_recode),type(train_losslist))
# print(range(0,epoch,interval_recode),type(train_losslist[0]))
# print(type(train_losslist[5]))
plt.plot(range(len(train_losslist)),train_losslist,label='TrainLoss')
plt.plot(range(len(train_losslist)),test_losslist,label='TestLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

plt.savefig(DIR+now+'loss')
plt.cla()

plt.plot(acclist,label='Test')
plt.hlines(1/18,-1,len(acclist)+1,colors='m',label='Expected')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.savefig(DIR+now+'accuracy')
