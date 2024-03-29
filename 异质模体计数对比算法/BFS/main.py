import networkx as nx
import time
import pandas as pd

# motif =

G = nx.read_edgelist("Caltech36.txt", create_using=nx.Graph())
info = pd.read_csv('Caltech36_info.csv',index_col=0)  # 读取节点信息



def isFeasibility(G, G_motif, G_node_list, n, G2_node_list, n1, MS, deep, info):
    for i in range(deep):
        if G_motif.has_edge(G2_node_list[i], G2_node_list[deep]) != G.has_edge(G_node_list[MS[i]],
                                                                               G_node_list[MS[deep]]):
            return False
        if G_motif.has_edge(G2_node_list[deep], G2_node_list[i]) != G.has_edge(G_node_list[MS[deep]],
                                                                               G_node_list[MS[i]]):
            return False
        if G_motif.degree(G2_node_list[deep]) > G.degree(G_node_list[MS[deep]]):
            return False
    if info._get_value(int(G_node_list[1]), 'info') == info._get_value(int(G_node_list[2]), 'info'):
        return False
    return True

def BFS1(G,motif=None,node=None):
    sss = 0
    nei_list = list(G.neighbors(node))
    for i, node1 in enumerate(nei_list):
        for j in range(i+1, len(nei_list)):

            if G.has_edge(nei_list[j], node1) == False and info._get_value(int(node1),
                                                                           'info1') != info._get_value(
                int(nei_list[j]), 'info1'):
                    sss += 1
    return sss


def BFS2(G, motif=None, node=None):
    sss = 0
    nei_list = list(G.neighbors(node))
    for i, node1 in enumerate(nei_list):
        for j in range(i + 1, len(nei_list)):

            if G.has_edge(nei_list[j], node1) == False and info._get_value(int(node1),
                                                                           'info1') == info._get_value(
                int(nei_list[j]), 'info1') and info._get_value(int(node1),'info1') == 1:
                sss += 1
    return sss

def BFS3(G, motif=None, node=None):
    sss = 0
    nei_list = list(G.neighbors(node))
    for i, node1 in enumerate(nei_list):
        for j in range(i + 1, len(nei_list)):

            if G.has_edge(nei_list[j], node1) == False and info._get_value(int(node1),
                                                                           'info1') == info._get_value(
                int(nei_list[j]), 'info1') and info._get_value(int(node1),'info1') == 0:
                sss += 1
    return sss


start = time.time()
for node in G.nodes():
    motif = nx.from_edgelist([(1, 2), (1, 3)], create_using=nx.Graph())
    s = BFS1(G, motif, node)
    b = BFS2(G, motif, node)
    bs = BFS3(G, motif, node)

print(time.time() - start)