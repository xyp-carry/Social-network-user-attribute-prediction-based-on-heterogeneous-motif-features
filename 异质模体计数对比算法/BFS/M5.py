import networkx as nx
import time
import pandas as pd

# motif =

G = nx.read_edgelist("Swarthmore42.txt", create_using=nx.Graph())
info = pd.read_csv('Swarthmore42_info.csv', index_col=0)  # 读取节点信息


def BFS1(G, motif=None, node=None):
    a = {}
    a['q1'] = 0
    a['q2'] = 0
    a['q3'] = 0
    a['q4'] = 0
    a['q5'] = 0
    a['q6'] = 0
    a['q7'] = 0
    a['q8'] = 0
    G2_node_list = list(nx.nodes(motif))
    n1 = len(G2_node_list)
    MS = [-1 for x in range(n1)]
    nei_list = list(G.neighbors(node))
    B_MS = []
    for i, node1 in enumerate(nei_list):
        if info._get_value(int(node1), 'info1') != 0:
            continue
        for j in G.neighbors(node1):
            if j == node1:
                continue
            if info._get_value(int(j), 'info1') != 0:
                continue
            B_MS.append(j)
        for z in B_MS:
            for zz in G.neighbors(z):
                if zz == node1:
                    continue
                if info._get_value(int(zz), 'info1') != 1:
                    a['q3'] += 1
                else:
                    a['q4'] += 1
    for i, node1 in enumerate(nei_list):
        if info._get_value(int(node1), 'info1') != 1:
            continue
        for j in G.neighbors(node1):
            if j == node1:
                continue
            if info._get_value(int(j), 'info1') != 0:
                continue
            B_MS.append(j)
        for z in B_MS:
            for zz in G.neighbors(z):
                if zz == node1:
                    continue
                if info._get_value(int(zz), 'info1') != 1:
                    a['q5'] += 1
                else:
                    a['q6'] += 1
    for z in B_MS:
        for zz in G.neighbors(z):
            if zz == node1:
                continue
            if info._get_value(int(zz), 'info1') != 1:
                a['q7'] += 1
            else:
                a['q8'] += 1
    for i, node1 in enumerate(nei_list):
        if info._get_value(int(node1), 'info1') != 1:
            continue
        for j in G.neighbors(node1):
            if j == node1:
                continue
            if info._get_value(int(j), 'info1') != 1:
                continue
            B_MS.append(j)
        for z in B_MS:
            for zz in G.neighbors(z):
                if zz == node1:
                    continue
                if info._get_value(int(zz), 'info1') != 1:
                    a['q1'] += 1
                else:
                    a['q2'] += 1
    return 1
def DFS(G,node=None):
    a = {}
    a['111'] = 0
    a['121'] = 0
    a['112'] = 0
    a['222'] = 0
    a['122'] = 0
    a['212'] = 0
    a['211'] = 0
    a['221'] = 0
    G2_node_list = list(nx.nodes(motif))
    n1 = len(G2_node_list)
    MS = [-1 for x in range(n1)]
    nei_list = list(G.neighbors(node))
    B_MS = []
    for i, node1 in enumerate(nei_list):
        if info._get_value(int(node1),'info1') == 0:
            continue
        for j in G.neighbors(node1):
            if info._get_value(int(j), 'info1') == 0:
                continue
            if j == node:
                continue
            for z in G.neighbors(j):
                if info._get_value(int(z), 'info1') == 0:
                    continue
                if z == node or z== node1:
                    continue
                a[str(info._get_value(int(node1),'info1'))+str(info._get_value(int(j),'info1'))+str(info._get_value(int(z),'info1'))]+=1


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
    motif = nx.from_edgelist([(1, 2), (1, 3), (1, 4)], create_using=nx.Graph())
    # s = BFS1(G, motif, node)
    DFS(G,node)
    # b, b2 = BFS2(G, motif, node)
    # bs = BFS3(G, motif, node)

print(time.time() - start)
