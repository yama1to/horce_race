

import pandas as pd 

data = 'csv/data_race_tokyo.csv'
data2 = 'csv/data4.csv'
# data2 = 'csv/data4.csv'
df = pd.read_csv(data, sep=",")

df2 = pd.read_csv(data2, sep=",")

# print(df.sort_index(axis=1).columns)
# print(df2.sort_index(axis=1).columns)


print(df2.columns)
print(df.columns)

len2 = len(df2.columns)
len1 = len(df.columns)

print(len2,len1)
# c=0
# for i in range(len1):

#     #if df2.columns[i] == df.columns[i]:
#         print(df2.columns[i],df.columns[i])

# print(c)
    #newdf = 1
# df2 = df2.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','is_tamer_same', 'is_owner_same','rank_1','pre_date_diff', 'is_rider_same',],axis=1)

# newdf = df2.sort_index(axis=1)#[:,:3]
# newdf.to_csv('csv/data4.csv')

# newdf = df.sort_index(axis=1)#[:,:3]
# newdf.to_csv('csv/data_race_tokyo.csv')