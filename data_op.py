import pandas as pd 
import numpy as np

df = pd.read_csv("./csv/data3.csv", sep=",")

# print(df.shape)
# print(df.columns)
# del_list = ['札幌','未勝利*','芝','良',,]
for i in range(len(df.columns)):
    print(i,df.columns[i])
# col = df.columns[45:-2]
# df = df.drop(col,axis= 1)
# df.to_csv('csv/data3.csv')
