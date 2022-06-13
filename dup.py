import pandas as pd 
import numpy as np

df = pd.read_csv("./csv/data2.csv", sep=",")
# print(df)
# df = pd.read_csv("./csv/data1.csv", sep=",")
"""
raceidの重複数をplot


"""

#print(df[df.duplicated()])

# df = df.drop_duplicates()
# df.to_csv("./csv/final_data_dup.csv", sep=",")
raceid = df['race_id']
idx = df.index
raceid_list2 = raceid.drop_duplicates()


# print(df)
# print(df.drop_duplicates())

l = np.zeros(len(raceid_list2))


raceid = raceid.to_numpy()
raceid_list = raceid_list2.to_numpy()
# print(raceid)
# print(raceid)c

import matplotlib.pyplot as plt 
from tqdm import tqdm 
x = 0
first = 201400000000
for i in tqdm(range(len(raceid))):
    # if raceid[i] > first:

    #     print(first)
    #     plt.hist(l)
    #     plt.show()
    #     first +=100000000
    #     l = np.zeros(len(raceid_list2))

    for j in range(x,len(raceid_list)):
        #print(raceid[1,i],raceid_list[1,j])
        if raceid[i] == raceid_list[j]:
            l[j] += 1
            x = j
            break
    
    
    
# print(l,np.max(l),np.min(l))

print(raceid)
plt.hist(l)
plt.show()


from pprint import pprint 
race1 = raceid_list[l!=16]
# race1 = raceid_list[raceid_list2]
# print(df.ix[0,1])
pprint(race1)


import multiprocessing as mp
#pool_obj = mp.Pool(17)


# import func
#for i in tqdm(range(len(race1))):
#pool_obj.map(func.func,range(len(race1)))
parallel = 20
# del_list= []
# def func(df,race1,del_list):
#     #df1 =df
#     #df = df.drop(index = df.index[race1 == df["race_id"]].tolist())
#     del_list.append(df.index[race1 == df["race_id"]].tolist())
#     print(len(del_list))
#     # print(df[race1[i] == df["race_id"]].index)

# for i in tqdm(range(0,len(race1)//16,16)):
#     print(i)
#     processes = [
#     mp.Process(target=func, args=(df,race1[i],del_list)) for i in range(i,i+16)
#     ]
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()
#df = df.drop(index = del_list)

for i in tqdm(range(len(race1))):
    #print(df.index[race1[i] == df["race_id"]].tolist())
    df = df.drop(index = (df.index[race1[i] == df["race_id"]].tolist()))
    print(df.info())
df.to_csv('./csv/data3.csv')


