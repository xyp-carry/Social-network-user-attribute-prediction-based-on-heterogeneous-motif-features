import os
import pandas as pd
import networkx as nx
import random
import itertools
import copy
from itertools import chain
# import xgboost as xgbc
import matplotlib as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
# from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

import Search

import time

"""=======================读取网络数据与节点信息数据（txt、csv）==========================="""
g = nx.read_edgelist('twi20.txt', nodetype=int)  # 读取网络信息
# info = pd.read_csv('Caltech36_info.csv')  # 读取节点信息
train = pd.read_csv('twi20_train_info.csv')
test = pd.read_csv('twi20_test_info.csv')
# test['info'] = test['info'].astype(int)
val = pd.read_csv('twi20_val_info.csv')
new = pd.concat([train, test,val], ignore_index=True)
# new = pd.concat([new, ], ignore_index=True)
info = new
# print(new._get_value(0, 'info'))
"""=======================数据预处理===========================
利用pandas的分组功能将数据集中的节点分为AB两类，非AB类保存至dl_node中并在网络中进行删除
目前仅设计了直属命名的筛选方式，并不智能化，若继续优化算法可以考虑优化这一块的内容
"""
# male = pd.DataFrame(columns={'node'})  # 构建A类表
# female = pd.DataFrame(columns={'node'})  # 构建B类表
# dl_node = pd.DataFrame()  # 构建删除表
# info_group = info.groupby('info')  # 分组
# for i in info_group.groups:
#     if i == 1:
#         male = pd.DataFrame(data=info_group.get_group(i)['num'])
#         male.columns = {'node'}
#     elif i == 2:
#         female = pd.DataFrame(data=info_group.get_group(i)['num'])
#         female.columns = {'node'}
#     else:
#         dl_node = pd.concat([dl_node, info_group.get_group(i)['num']])
# dl_node.columns = {'node'}
#
# for i in dl_node['node']:
#     g.remove_node(i)
g = g.to_undirected()  # 剔除无用节点并转换为无向无权图

# for i in train.index:
#     if train['num'][i] not in g.nodes():
#         print(i)
#         train = train.drop(i)

# for i in test.index:
#     if test['num'][i] not in list(g.nodes()):
#         print(i)
#         test = test.drop(i)
#
# for i in val.index:
#     if val['num'][i] not in list(g.nodes()):
#         print(i)
#         val = val.drop(i)

"""=======================定义点模体结构==========================="""


def m4_1():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (1, 3), (1, 4)])
    motif = motif.to_undirected()
    return motif


def m4_2():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (2, 3), (2, 4)])
    motif = motif.to_undirected()
    return motif


def m4_3():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (1, 3), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m4_4():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (2, 3), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m4_5():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 3), (1, 2), (2, 4), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m4_6():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m4_7():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m4_8():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m4_9():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
    motif = motif.to_undirected()
    return motif


def m4_10():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 3), (1, 4), (3, 4), (2, 3), (2, 4)])
    motif = motif.to_undirected()
    return motif


def m4_11():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3, 4])
    motif.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    motif = motif.to_undirected()
    return motif


def m3_1():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3])
    motif.add_edges_from([(1, 2), (2, 3)])
    motif = motif.to_undirected()
    return motif


def m3_2():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3])
    motif.add_edges_from([(1, 2), (1, 3)])
    motif = motif.to_undirected()
    return motif


def m3_3():
    motif = nx.DiGraph()
    motif.add_nodes_from([1, 2, 3])
    motif.add_edges_from([(1, 2), (1, 3), (2, 3)])
    motif = motif.to_undirected()
    return motif


"""=========================定义点模体结构==========================="""

motif_structure = ['M4_5','M4_1', 'M4_2', 'M4_3', 'M4_4' , 'M4_6', 'M4_7', 'M4_8', 'M4_9', 'M4_10', 'M4_11', 'M3_1',
                   'M3_2', 'M3_3']
motif_label = ['111', '112', '121', '122', '211', '212', '221', '222']
motif_label3 = ['11', '12', '21', '22']

numb_motif = {}
numb_motif3 = {}
for i in motif_label:
    numb_motif[i] = 0
for i in motif_label3:
    numb_motif3[i] = 0

train_g = copy.deepcopy(g)
predictor_edge_value_train = []

"""=========================A样本特征提取==========================="""
a = 0
train_nodes = train['num']
for predictor in motif_structure:
    a = a + 1
    if a > 11:
        print('训练')
        print(predictor)
        predictor = predictor.lower()
        edge_value1 = []  # 每条边得分
        edge_value2 = []
        edge_value3 = []
        edge_value4 = []
        edge_value5 = []
        edge_value = [edge_value1, edge_value2, edge_value3, edge_value4, edge_value5]
        for node in train_nodes:  # 训练集的点
            for i in motif_label3:
                numb_motif3[i] = 0
            test_g = globals().get(predictor)()  # 训练集网络
            if node not in g.nodes():
                1
            else:
                motif_num = Search.node_motif_num(g, test_g, node, numb_motif3, predictor.upper(), info)
            # motif_num = motifNum8.edge_motif_num(g, test_g, (u, v), numb_motif3, predictor, info, False, False)
            # globals() 函数会以字典类型返回当前位置的全部全局变量。 get函数返回指定的键
            for i in range(4):  # 得到一个边在一个预测器的所有评分
                edge_value[i].append(numb_motif3[motif_label3[i]])
            numb_motif3[predictor] = 0
            for i in range(4):
                numb_motif3[predictor] = numb_motif3[predictor] + numb_motif3[motif_label3[i]]
            edge_value[4].append(numb_motif3[predictor])
            numb_motif3.pop(predictor)
            end3 = time.time()

        for i in range(5):
            predictor_edge_value_train.append(edge_value[i])  # 遍历所有预测器得出
        # 得分
    else:
        print('训练')
        print(predictor)
        predictor = predictor.lower()
        edge_value1 = []  # 每条边得分
        edge_value2 = []
        edge_value3 = []
        edge_value4 = []
        edge_value5 = []
        edge_value6 = []
        edge_value7 = []
        edge_value8 = []
        edge_value9 = []
        edge_value = [edge_value1, edge_value2, edge_value3, edge_value4, edge_value5, edge_value6, edge_value7,
                      edge_value8, edge_value9]
        start3 = time.time()
        for node in train_nodes:  # 训练集的边
            for i in motif_label:
                numb_motif[i] = 0
            test_g = globals().get(predictor)()  # 训练集网络
            if node not in g.nodes():
                1
            else:
                motif_num = Search.node_motif_num(g, test_g, node, numb_motif, predictor.upper(), info)
            # motif_num = motifNum8.edge_motif_num(g, test_g, (u, v), numb_motif, predictor, info, False, False)
            # globals() 函数会以字典类型返回当前位置的全部全局变量。 get函数返回指定的键
            for i in range(8):  # 得到一个边在一个预测器的所有评分
                edge_value[i].append(numb_motif[motif_label[i]])
            numb_motif[predictor] = 0
            for i in range(8):
                numb_motif[predictor] = numb_motif[predictor] + numb_motif[motif_label[i]]
            edge_value[8].append(numb_motif[predictor])
            numb_motif.pop(predictor)
            end3 = time.time()

        for i in range(9):
            predictor_edge_value_train.append(edge_value[i])  # 遍历所有预测器得出
        # 得分

df_train = pd.DataFrame()
start_1 = train_nodes
df_train['src'] = start_1

c = 0
a = 0
for i in range(len(motif_structure)):

    a = a + 1
    if a > 11:
        for j in range(len(motif_label3)):
            df_train[motif_structure[i] + motif_label3[j]] = predictor_edge_value_train[c]
            c = c + 1
        df_train[motif_structure[i]] = predictor_edge_value_train[c]
        c += 1
    else:
        for j in range(len(motif_label)):
            df_train[motif_structure[i] + motif_label[j]] = predictor_edge_value_train[c]
            c += 1
        df_train[motif_structure[i]] = predictor_edge_value_train[c]
        c += 1

df_train['label'] = train['info']  # 标签为0代表负样本，标签为1代表正样本
df_train.to_csv('twi_train_2.csv', index=False)

"""=========================B样本特征提取==========================="""
predictor_edge_value_train = []
a = 0
train_nodes = test['num']
for predictor in motif_structure:
    a = a + 1
    if a > 11:
        print('训练')
        print(predictor)
        predictor = predictor.lower()
        edge_value1 = []  # 每条边得分
        edge_value2 = []
        edge_value3 = []
        edge_value4 = []
        edge_value5 = []
        edge_value = [edge_value1, edge_value2, edge_value3, edge_value4, edge_value5]
        for node in train_nodes:  # 训练集的点
            for i in motif_label3:
                numb_motif3[i] = 0
            test_g = globals().get(predictor)()  # 训练集网络
            if node not in g.nodes():
                1
            else:
                motif_num = Search.node_motif_num(g, test_g, node, numb_motif3, predictor.upper(), info)
            # motif_num = motifNum8.edge_motif_num(g, test_g, (u, v), numb_motif3, predictor, info, False, False)
            # globals() 函数会以字典类型返回当前位置的全部全局变量。 get函数返回指定的键
            for i in range(4):  # 得到一个边在一个预测器的所有评分
                edge_value[i].append(numb_motif3[motif_label3[i]])
            numb_motif3[predictor] = 0
            for i in range(4):
                numb_motif3[predictor] = numb_motif3[predictor] + numb_motif3[motif_label3[i]]
            edge_value[4].append(numb_motif3[predictor])
            numb_motif3.pop(predictor)
            end3 = time.time()

        for i in range(5):
            predictor_edge_value_train.append(edge_value[i])  # 遍历所有预测器得出
        # 得分
    else:
        print('训练')
        print(predictor)
        predictor = predictor.lower()
        edge_value1 = []  # 每条边得分
        edge_value2 = []
        edge_value3 = []
        edge_value4 = []
        edge_value5 = []
        edge_value6 = []
        edge_value7 = []
        edge_value8 = []
        edge_value9 = []
        edge_value = [edge_value1, edge_value2, edge_value3, edge_value4, edge_value5, edge_value6, edge_value7,
                      edge_value8, edge_value9]
        start3 = time.time()
        for node in train_nodes:  # 训练集的边
            for i in motif_label:
                numb_motif[i] = 0
            test_g = globals().get(predictor)()  # 训练集网络
            if node not in g.nodes():
                1
            else:
                motif_num = Search.node_motif_num(g, test_g, node, numb_motif, predictor.upper(), info)
            # motif_num = motifNum8.edge_motif_num(g, test_g, (u, v), numb_motif, predictor, info, False, False)
            # globals() 函数会以字典类型返回当前位置的全部全局变量。 get函数返回指定的键
            for i in range(8):  # 得到一个边在一个预测器的所有评分
                edge_value[i].append(numb_motif[motif_label[i]])
            numb_motif[predictor] = 0
            for i in range(8):
                numb_motif[predictor] = numb_motif[predictor] + numb_motif[motif_label[i]]
            edge_value[8].append(numb_motif[predictor])
            numb_motif.pop(predictor)
            end3 = time.time()

        for i in range(9):
            predictor_edge_value_train.append(edge_value[i])  # 遍历所有预测器得出
        # 得分

df_train = pd.DataFrame()
start_1 = train_nodes
df_train['src'] = start_1

c = 0
a = 0
for i in range(len(motif_structure)):

    a = a + 1
    if a > 11:
        for j in range(len(motif_label3)):
            df_train[motif_structure[i] + motif_label3[j]] = predictor_edge_value_train[c]
            c = c + 1
        df_train[motif_structure[i]] = predictor_edge_value_train[c]
        c += 1
    else:
        for j in range(len(motif_label)):
            df_train[motif_structure[i] + motif_label[j]] = predictor_edge_value_train[c]
            c += 1
        df_train[motif_structure[i]] = predictor_edge_value_train[c]
        c += 1

df_train['label'] = test['info']  # 标签为0代表负样本，标签为1代表正样本
df_train.to_csv('twi_test_2.csv', index=False)


"""=========================C样本特征提取==========================="""
predictor_edge_value_train = []
a = 0
train_nodes = val['num']
for predictor in motif_structure:
    a = a + 1
    if a > 11:
        print('训练')
        print(predictor)
        predictor = predictor.lower()
        edge_value1 = []  # 每条边得分
        edge_value2 = []
        edge_value3 = []
        edge_value4 = []
        edge_value5 = []
        edge_value = [edge_value1, edge_value2, edge_value3, edge_value4, edge_value5]
        for node in train_nodes:  # 训练集的点
            for i in motif_label3:
                numb_motif3[i] = 0
            test_g = globals().get(predictor)()  # 训练集网络
            if node not in g.nodes():
                1
            else:
                motif_num = Search.node_motif_num(g, test_g, node, numb_motif3, predictor.upper(), info)
            # motif_num = motifNum8.edge_motif_num(g, test_g, (u, v), numb_motif3, predictor, info, False, False)
            # globals() 函数会以字典类型返回当前位置的全部全局变量。 get函数返回指定的键
            for i in range(4):  # 得到一个边在一个预测器的所有评分
                edge_value[i].append(numb_motif3[motif_label3[i]])
            numb_motif3[predictor] = 0
            for i in range(4):
                numb_motif3[predictor] = numb_motif3[predictor] + numb_motif3[motif_label3[i]]
            edge_value[4].append(numb_motif3[predictor])
            numb_motif3.pop(predictor)
            end3 = time.time()

        for i in range(5):
            predictor_edge_value_train.append(edge_value[i])  # 遍历所有预测器得出
        # 得分
    else:
        print('训练')
        print(predictor)
        predictor = predictor.lower()
        edge_value1 = []  # 每条边得分
        edge_value2 = []
        edge_value3 = []
        edge_value4 = []
        edge_value5 = []
        edge_value6 = []
        edge_value7 = []
        edge_value8 = []
        edge_value9 = []
        edge_value = [edge_value1, edge_value2, edge_value3, edge_value4, edge_value5, edge_value6, edge_value7,
                      edge_value8, edge_value9]
        start3 = time.time()
        for node in train_nodes:  # 训练集的边
            for i in motif_label:
                numb_motif[i] = 0
            test_g = globals().get(predictor)()  # 训练集网络
            if node not in g.nodes():
                1
            else:
                motif_num = Search.node_motif_num(g, test_g, node, numb_motif, predictor.upper(), info)
            # motif_num = motifNum8.edge_motif_num(g, test_g, (u, v), numb_motif, predictor, info, False, False)
            # globals() 函数会以字典类型返回当前位置的全部全局变量。 get函数返回指定的键
            for i in range(8):  # 得到一个边在一个预测器的所有评分
                edge_value[i].append(numb_motif[motif_label[i]])
            numb_motif[predictor] = 0
            for i in range(8):
                numb_motif[predictor] = numb_motif[predictor] + numb_motif[motif_label[i]]
            edge_value[8].append(numb_motif[predictor])
            numb_motif.pop(predictor)
            end3 = time.time()

        for i in range(9):
            predictor_edge_value_train.append(edge_value[i])  # 遍历所有预测器得出
        # 得分

df_train = pd.DataFrame()
start_1 = train_nodes
df_train['src'] = start_1

c = 0
a = 0
for i in range(len(motif_structure)):

    a = a + 1
    if a > 11:
        for j in range(len(motif_label3)):
            df_train[motif_structure[i] + motif_label3[j]] = predictor_edge_value_train[c]
            c = c + 1
        df_train[motif_structure[i]] = predictor_edge_value_train[c]
        c += 1
    else:
        for j in range(len(motif_label)):
            df_train[motif_structure[i] + motif_label[j]] = predictor_edge_value_train[c]
            c += 1
        df_train[motif_structure[i]] = predictor_edge_value_train[c]
        c += 1

df_train['label'] = val['info']  # 标签为0代表负样本，标签为1代表正样本
df_train.to_csv('twi_val_2.csv', index=False)