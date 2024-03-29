import networkx as nx
import time
import pandas as pd

# motif =

G = nx.read_edgelist("Caltech36.txt", create_using=nx.Graph())
info = pd.read_csv('Caltech36_info.csv', index_col=0)  # 读取节点信息


def BFS(G, MS, deep=-1, search=-1):
    MAXLEN = len(MS)
    MINLEN = search
    for c in range(MINLEN, MAXLEN):
        search += 1
        for nn in G.neighbors(MS[c]):
            if nn not in MS:
                MS.append(nn)
                deep += 1
    if search == deep:
        return
    else:
        BFS(G, MS, deep, search)


def BFS1(G, motif=None, node=None):
    sss = 0
    ccc = 0
    qqq = 0
    ppp = 0
    ma = 0
    me = 0
    G2_node_list = list(nx.nodes(motif))
    n1 = len(G2_node_list)
    MS = []
    deep = 0

    return sss


def BFS2(G, motif=None, node=None):
    sss, ccc = 0, 0
    nei_list = list(G.neighbors(node))
    for i, node1 in enumerate(nei_list):
        for j in range(i + 1, len(nei_list)):
            for z in range(j + 1, len(nei_list)):
                if G.has_edge(nei_list[j], node1) == False and G.has_edge(node1, nei_list[z]) and G.has_edge(
                        nei_list[z], nei_list[j]):
                    if info._get_value(int(node1), 'info1') == info._get_value(int(nei_list[j]),
                                                                               'info1') and info._get_value(
                        int(node1), 'info1') == info._get_value(int(nei_list[z]), 'info1'):
                        if info._get_value(int(node1), 'info1') == 1:
                            sss += 1
                        else:
                            ccc += 1
    return sss, ccc


def BFS3(G, motif=None, node=None):
    sss = 0
    nei_list = list(G.neighbors(node))
    for i, node1 in enumerate(nei_list):
        for j in range(i + 1, len(nei_list)):

            if G.has_edge(nei_list[j], node1) == False and info._get_value(int(node1),
                                                                           'info1') == info._get_value(
                int(nei_list[j]), 'info1') and info._get_value(int(node1), 'info1') == 0:
                sss += 1
    return sss


start = time.time()
for node in G.nodes():
    motif = nx.from_edgelist([(1, 9), (1, 3), (1, 4), (2, 9)], create_using=nx.Graph())
    MS = [1]
    BFS3(G, motif, node)

    # s = BFS1(G, motif, node)
    # b, b2 = BFS2(G, motif, node)
    # bs = BFS3(G, motif, node)

print(time.time() - start)
