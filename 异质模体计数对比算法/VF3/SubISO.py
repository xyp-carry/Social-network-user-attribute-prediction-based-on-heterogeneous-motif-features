# -*- coding: utf-8 -*-
"""
Created on Thurs March 18 14:20 2021

@author: WangJiawei
"""

import networkx as nx
import threading
import time
import os
from multiprocessing import Process,Pool,cpu_count


class motif:
    """
    无向网络模体类
    """
    def __init__(self):
        self.M3=list()
        self.M3.append(nx.from_edgelist([(1, 2), (1, 3)], create_using=nx.Graph()))
        # self.M3.append(nx.from_edgelist([(1, 2), (1, 3),(2,3)], create_using=nx.Graph()))
        self.M3_num=2
        self.M4=list()
        # self.M4.append(nx.from_edgelist([(1, 2), (1, 3), (1, 4)], create_using=nx.Graph()))
        self.M4.append(nx.from_edgelist([(1, 2), (2, 3), (3, 4)], create_using=nx.Graph()))
        # self.M4.append(nx.from_edgelist([(1, 2), (2, 3), (3, 4),(4,1)], create_using=nx.Graph()))
        # self.M4.append(nx.from_edgelist([(1, 2), (1, 3), (2, 3),(3,4)], create_using=nx.Graph()))
        # self.M4.append(nx.from_edgelist([(1, 2), (1, 3), (2, 3),(2,4),(3,4)], create_using=nx.Graph()))
        # self.M4.append(nx.from_edgelist([(1, 2), (1, 3), (2, 3),(2,4),(3,4),(1,4)], create_using=nx.Graph()))
        self.M4_num=6



def isFeasibility(G,G_motif,G_node_list,n,G2_node_list,n1,MS,deep):
    for i in range(deep):
        if G_motif.has_edge(G2_node_list[i],G2_node_list[deep]) != G.has_edge(G_node_list[MS[i]],G_node_list[MS[deep]]):
            return False
        if G_motif.has_edge(G2_node_list[deep],G2_node_list[i]) != G.has_edge(G_node_list[MS[deep]],G_node_list[MS[i]]):
            return False
        if G_motif.degree(G2_node_list[deep]) > G.degree(G_node_list[MS[deep]]):
            return False
    return True


def dfs_motif(G,G_motif,G_node_list,n,G2_node_list,n1,MS,status,deep,result,path_lenG,length):

    if deep >= n1:
        result[MS[0]] += 1           #约定status存的是序号，查找的时候用  G_node_list[MS[i]]
        return 1
    sum=0
    for i in range(n):
        if i in MS[0:deep]:
            continue
        if deep !=0 :
            # print("+++++",type(G_node_list[i]))
            # print("_____________",type(G_node_list[MS[0]]))
            # print("WWWWWW",path_lenG['2']['1'])
            # print('sssssssssssss',path_lenG[G_node_list[i]][G_node_list[MS[0]]])

            if path_lenG[int(G_node_list[i])][int(G_node_list[MS[0]])] > length:
                continue

        MS[deep]=i
        if(isFeasibility(G,G_motif,G_node_list,n,G2_node_list,n1,MS,deep)):
            sum+=dfs_motif(G,G_motif,G_node_list,n,G2_node_list,n1,MS,status,deep+1,result,path_lenG,length)

    return sum

def center_node(G, G_motif):
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    status = []
    n = len(G_node_list)
    n1 = len(G2_node_list)
    path_lenM1 = dict(nx.all_pairs_shortest_path_length(G_motif))
    path_lenM = {}
    for i in range(n1):
        path_lenM[int(G2_node_list[i])] = {}
        for j in range(n1):
            path_lenM[int(G2_node_list[i])][int(G2_node_list[j])] = 999999

    for (i, a) in path_lenM1.items():
        for (j, k) in a.items():
            path_lenM[int(i)][int(j)] = k


    path_lenG1 = dict(nx.all_pairs_shortest_path_length(G))

    path_lenG = {}
    for i in range(n):
        path_lenG[int(G_node_list[i])] = {}
        for j in range(n):
            path_lenG[int(G_node_list[i])][int(G_node_list[j])] = 999999

    for (i, a) in path_lenG1.items():
        for (j, k) in a.items():
            path_lenG[int(i)][int(j)] = k

    length = 999999
    k = 0
    for i in range(n1):
        s = 0
        for j in range(n1):
            if i != j and path_lenM[G2_node_list[i]][G2_node_list[j]] > s:
                s = path_lenM[G2_node_list[i]][G2_node_list[j]]
        if s < length:
            length = s
            k = i
    return k,length,path_lenG,path_lenM

def total_motif_num(G,G_motif):
    G_node_list = list(nx.nodes(G))

    G2_node_list = list(nx.nodes(G_motif))
    status = []
    n = len(G_node_list)
    n1 = len(G2_node_list)
    G1=nx.to_undirected(G)
    G_motif1=nx.to_undirected(G_motif)
    k, length, path_lenG,path_lenM =center_node(G1,G_motif1)

    temp = G2_node_list[0]
    G2_node_list[0]=G2_node_list[k]
    G2_node_list[k]=temp


    MS = [-1 for x in range(n1)]
    result = [0 for x in range(n)]
    sum=dfs_motif(G, G_motif, G_node_list, n, G2_node_list, n1, MS, status, 0, result,path_lenG,length)
    a = dfs_motif(G_motif, G_motif, G2_node_list, n1, G2_node_list, n1, MS, status, 0, result,path_lenM,length)

    return sum/a


def node_motif_num(G,G_motif):
    G_node_list= list(nx.nodes(G))
    G2_node_list= list(nx.nodes(G_motif))
    status=[]
    n1=len(G2_node_list)
    n=len(G_node_list)
    print(n,n1)
    MS=[-1 for x in range(n1)]
    result=[0 for x in range(n)]
    ######################3

    start = time.time()

    sum=[]
    for i in range(n):
        MS[0]=i
        sum.append( dfs_motif(G, G_motif, G_node_list, n, G2_node_list, n1, MS, status, 1, result))
        # p=threading.Thread(target=vf2_motif,args=(G,G_motif,G_node_list,n,G2_node_list,n1,MS,status,1,result))
        ##由于MS会被改变所以无法使用多线程。

    end = time.time()
    print("总共用时{}秒".format((end - start)))
    ############33
    # sum=vf2_motif(G,G_motif,G_node_list,n,G2_node_list,n1,MS,status,0,result)
    return result

def jisuan(G, G_motif):
    sss = node_motif_num(G, G_motif)
    s = 0
    for i in sss:
        s += i
    print(s)
    print("____________________________")

    a = node_motif_num(G_motif, G_motif)
    print("____________________________")
    b = 0

    for i in a:
        b += i
    print(b)
    print(s / b)

def SubISO(G,motif):

    n=len(motif)
    result=[]
    for i in range(n):
        res=total_motif_num(G,motif[i])
        result.append(res)
    return result


if __name__ == "__main__":
    import pandas as pd
    import time
    # data = pd.read_csv("mydata/USAir97.txt", sep=" ", names=["a", "b", "w"])
    # data = pd.read_csv("mydata/IEEE300.txt", sep="\s", names=["a", "b"])
    # G = nx.from_pandas_edgelist(data, source="a", target="b", create_using=nx.Graph())
    G = nx.read_edgelist("Caltech36.txt", create_using=nx.Graph())
    node_list=list(G.nodes)

    edge_list = list(nx.edges(G))
    # print(edge_list)
    print("node :", len(node_list), "edge :", len(edge_list))
    starttime = time.time()
    Moti =motif()
    b=SubISO(G,Moti.M4)
    print("————————",b)
    endtime = time.time()
    print('all motif time ', endtime - starttime)
