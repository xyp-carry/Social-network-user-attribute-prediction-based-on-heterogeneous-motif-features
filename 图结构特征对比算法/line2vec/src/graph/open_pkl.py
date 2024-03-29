import pickle
data  = open('edge_map.pkl','rb')
data = pickle.load(data)
nf = open('myDictionary.pkl','rb')
data1 = pickle.load(nf)
# print(data1)
print(data)

# for m in range(142):
#     for e in data:
#         if data[e] == m:
#             print(data1[e])