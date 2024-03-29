import networkx as nx
from networkx.algorithms import node_classification
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import random
import copy

def rand_edge(G, vi, vj, p=0.1):  # 默认概率p=0.1
    probability = random.random()  # 生成随机小数
    if (probability < p):  # 如果小于p
        G.add_edge(vi, vj)


def create_toroidal(k):
    G = nx.DiGraph()
    # G=nx.Graph()
    count = 0
    mapping1 = {}
    mapping2 = {}
    for i in range(k):
        G.add_node(i)
    for i in range(k):
        for j in range(i, k):
            rand_edge(G, i, j)
            rand_edge(G, j, i)
    return G


def your_harmonic_function(Graph, label_name):
    # YOUR CODE HERE
    x = nx.spring_layout(G, iterations=1)
    existing_key = 1
    dims = len(x[existing_key])
    weights = {}
    sigma = []
    for i in range(dims):
        max_d = 0
        for edge in Graph.edges:
            if (x[edge[0]][i] - x[edge[1]][i]) ** 2 > max_d:
                max_d = (x[edge[0]][i] - x[edge[1]][i]) ** 2
        sigma.append(max_d)
    for edge in Graph.edges:
        weight = 0
        for i in range(dims):
            weight += ((x[edge[0]][i] - x[edge[1]][i]) ** 2) / sigma[i]
        weights[edge] = math.exp(-weight)
        weights[(edge[1],edge[0])] = 1-weights[edge]
    fu = {}
    fl = {}
    uniq_labeles = {}
    uniq_labeles_n = 0
    uniq_labeles_inverse = []
    ccc = list(Graph.nodes)
    ccc.sort()
    for key in ccc:
        try:
            label = Graph.nodes[key][label_name]
            if label not in uniq_labeles:
                uniq_labeles[label] = uniq_labeles_n
                uniq_labeles_inverse.append(label)
                uniq_labeles_n += 1
            fl[key] = uniq_labeles[label]
        except KeyError:
            fu[key] = ""
    print(uniq_labeles)
    fu_list = [key for key in fu]
    fl_list = [key for key in fl]
    print(fu)
    print(fu_list)
    print(fl_list)
    Duu = np.zeros((len(fu), len(fu)))
    Wuu = np.zeros((len(fu), len(fu)))
    Wul = np.zeros((len(fu), len(fl)))
    row = 0
    for u_index in fu_list:
        column = 0
        for l_index in fl_list:
            try:
                Wul[row][column] = weights[(u_index, l_index)]
            except KeyError:
                pass
            column += 1
        column = 0
        for u_index2 in fu_list:
            try:
                Wuu[row][column] = weights[(u_index, u_index2)]
            except KeyError:
                pass
            column += 1
        row += 1
    for row in range(len(fu)):
        sum = 0
        for column in range(len(fu)):
            sum += Wuu[row][column]
        for column in range(len(fl)):
            sum += Wul[row][column]
        Duu[row][row] = sum
    fl_values = np.array([fl[key] for key in fl_list])

    print(np.matrix(Duu),np.matrix(Wuu),np.matrix(Wul))

    print(Duu, Wuu, Wul)
    ccc = Duu - Wuu
    fu_values = np.dot(np.dot(np.linalg.inv(Duu - Wuu), Wul), fl_values)
    print(min(fu_values))
    for row in range(len(fu_values)):
        fu[fu_list[row]] = fu_values[row]
    labeled_list = []
    # for i in range(len(fu) + len(fl) + 1):
    #     if i == 0:
    #         continue
    #     try:
    #         labeled_list.append(fl[i])
    #     except KeyError:
    #         try:
    #             labeled_list.append(fu[i])
    #         except KeyError:
    #             continue
    for i in range(len(fu) + len(fl) + 1):
        if i == 0:
            continue
        try:
            labeled_list.append(fu[i])
        except KeyError:
            try:
                1
            except KeyError:
                continue

    return [uniq_labeles_inverse[int(round(value))] for value in labeled_list]


G = create_toroidal(256)
G = nx.read_edgelist('twi20.txt', nodetype=int)
print(G.is_directed())

G.nodes[1]['label'] = 'blue'
G.nodes[2]['label'] = 'red'
G.nodes[3]['label'] = 'blue'
G.nodes[4]['label'] = 'red'
# gpos = nx.spring_layout(G, iterations=200)
# node_color = ['blue' if n == 0 else 'red' if n == 255 else 'gray' for n in G.nodes]
# node_color = ['blue', 'red', 'blue', 'red']
# plt.figure(figsize=(10, 10))
# nx.draw(G, gpos, with_labels=False, node_size=200, node_color=node_color)
plt.show()
G_undirected = G.to_undirected()

color_list = []
a_train = pd.read_csv('twi20_train_info.csv', index_col='num')
a_val = pd.read_csv('twi20_val_info.csv',index_col='num')
a_test = pd.read_csv('twi20_test_info.csv',index_col='num')
#
for i in list(a_train['info']):
    if i == 1:
        color_list.append("blue")
    else:
        color_list.append("red")
for i in list(a_val['info']):
    if i == 1:
        color_list.append("blue")
    else:
        color_list.append("red")
for i in list(a_test['info']):
    if i == 1:
        color_list.append("blue")
    else:
        color_list.append("red")

print(len(color_list))

# node_color = node_classification.harmonic_function(G_undirected,20)
# print(node_color)


nodes = list(G.nodes)
nodes.sort()
print(nodes)
removed = [n for n in nodes if n % 4 == 0]
print(removed)
xx = copy.deepcopy(G)
# removed = [1, 2]

print(max(list(nx.connected_components(xx))[0]))

color_list_1 = []
for n in list(nx.connected_components(xx))[0]:
    if n not in removed:
        try:
            G.nodes[n]['label'] = color_list[n - 1]
            # print(list(G.neighbors(n)))
            if len(list(G.neighbors(n))) == 0:
                print(n)
                G.remove_node(n)
        except:
            print(n)
    else:
        color_list_1.append(color_list[n - 1])

for n in nx.connected_components(xx):
    if len(n) > 1000:
        print(1)
        continue
    for c in n:
        G.nodes[c]['label'] = color_list[c - 1]
print(n)
pre_color_list = []
predicted = your_harmonic_function(G, label_name='label')
# for n in list(nx.connected_components(xx))[0]:
#     if n in removed:
#         pre_color_list.append(predicted[n-1])
print(len(predicted))
print(len(pre_color_list))

# print(confusion_matrix(color_list, predicted))
#
# print(precision_recall_fscore_support(color_list, predicted))
