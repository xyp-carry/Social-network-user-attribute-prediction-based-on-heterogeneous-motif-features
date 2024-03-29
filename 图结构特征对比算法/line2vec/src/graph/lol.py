import pandas as pd
import random
data = pd.read_excel('test.xlsx')
df = pd.DataFrame()
c = len(data) * 0.05
mo = []
while(1):
    test =random.choice(data.index)
    if test in mo:
        continue
    # while(1):
    #     1
    if data['label'][test] == 1:
        df = df.append(data.loc[test],ignore_index=True)
        mo.append(test)
        # print('yes')
        c = c - 1
    if c < 0:
        break
c = len(data) * 0.05
while(1):
    test =random.choice(data.index)
    if test in mo:
        continue
    # while(1):
    #     1
    if data['label'][test] == 0:
        df = df.append(data.loc[test],ignore_index=True)
        mo.append(test)
        # print('yes')
        c = c - 1
    if c < 0:
        break
mo = sorted(mo)
mo.reverse()
print(type(mo))
for i in mo:
    print(i)
    data.drop(data.index[i], inplace=True)
df.to_csv('test.csv',index=None)
data.to_csv('train.csv',index=None)
