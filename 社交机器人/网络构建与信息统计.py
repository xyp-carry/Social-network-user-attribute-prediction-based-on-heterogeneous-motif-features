# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:46:12 2023

@author: Administrator
"""
import networkx as nx
import pandas as pd
import numpy as np
from numpy import random as nr
import igraph as ig

import matplotlib.pyplot as plt

# =============================================================================
# 读取数据
# =============================================================================
data = 'network1.csv'
df = pd.read_csv(data)
# df.info()
df['inviter_id'] = df['inviter_id'].astype('int') #格式转换
df['voter_id'] = df['voter_id'].astype('int')
# =============================================================================
# 生成网络
# =============================================================================
G = nx.from_pandas_edgelist(df,source='inviter_id',target='voter_id')
G = G.to_undirected()
nodes = G.nodes() #节点
edges_all = list(G.edges())
node_num,edge_num = len(nodes),len(edges_all)
print("######### n_num={},edge_num={} #########".format(node_num,edge_num))
for e in edges_all: #剔除自环边
    if e[0]==e[1]: edges_all.remove(e)
ree_flag = True if edge_num != len(edges_all) else False
print("G中是否有重复边: {}".format(ree_flag))

### 网络连通性和联通片信息 ##########
print("G的连通性：",nx.is_connected(G))
###### G中联通片数量
connected_components_number=nx.number_connected_components(G)
print("G中联通片数量：",connected_components_number)
# 获得最大联通片
argest_cc = max(nx.connected_components(G), key=len)
print("G中最大联通片节点数：",len(argest_cc))
### 获得最大联通
argest_G = G.subgraph(argest_cc)

### 网络最大联通片中平均最短路径长度和直径
Average_path_length=nx.average_shortest_path_length(argest_G)
D=nx.diameter(argest_G)
print("G中最大联通子图的平均最短路径长度=",Average_path_length)
print("G中最大联通子图的直径=",Average_path_length)



# =============================================================================
# 画图
# =============================================================================
pos=nx.spring_layout(argest_G)
nx.draw(argest_G,pos,with_labels=True)