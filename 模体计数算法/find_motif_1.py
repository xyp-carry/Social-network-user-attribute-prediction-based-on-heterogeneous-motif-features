import networkx as nx
import time
from collections import deque
import copy

def Search_M1_M4_M6(MS,SL,deep,deeps,excel,Motif_S,do_info,info,test,motif_numb):
    for i in SL[deep - deeps][0]:
        try:
            repeat = info._get_value(i - 1, 'info')
            if repeat == 0 or do_info[1] == 0 or do_info[2] == 0:
                continue
            if do_info[1] != do_info[2]:
                motif_numb[str(12) + str(repeat)] += 1
            else:
                if repeat == do_info[1]:
                    motif_numb[str(repeat) + str(repeat) + str(repeat)] += 1
                else:
                    if do_info[1] == 1:
                        motif_numb[str(121)] += 1
                    else:
                        motif_numb[str(122)] += 1
        except:
            cc = 1
        else:
            cc = 1

def Search_M1_1_M3_M4_1_M5_M3_2(MS,SL,deep,deeps,excel,Motif_S,do_info,info,test,motif_numb):
    # excel[(excel['node1']==MS[0])&(excel['node2']==MS[1])]+=len(SL[deep][0])
    # excel.at[excel.loc[excel['node1'] == MS[0]].loc[excel['node2'] == MS[1]].index.tolist()[0], Motif_S + str(1111)] += len(SL[deep][0])

    for i in SL[deep - deeps][0]:
        try:
            repeat = info._get_value(i - 1, 'info')
            if repeat == 0 or do_info[1] == 0 or do_info[2] == 0:
                continue
            if do_info[2] != repeat:
                motif_numb[str(do_info[1]) + str(12)] += 1
            else:
                if repeat == do_info[1]:
                    motif_numb[str(repeat) + str(repeat) + str(repeat)] += 1
                else:
                    if do_info[1] == 1:
                        motif_numb[str(122)] += 1
                    else:
                        motif_numb[str(211)] += 1
        except:
            cc = 1
        else:
            cc = 1

def Search_M2_M2_1_M3_1(MS,SL,deep,deeps,excel,Motif_S,do_info,info,test,motif_numb):
    for i in SL[deep - deeps][0]:
        try:
            repeat = info._get_value(i - 1, 'info')
            if repeat == 0 or do_info[1] == 0 or do_info[2] == 0:
                continue
            else:
                motif_numb[str(do_info[1]) + str(do_info[2]) + str(repeat)] += 1
        except:
            cc = 1
        else:
            cc = 1


#  excel.to_csv('test.csv',index= False)


def Search_M33_1_M33_2(MS, SL, deep, deeps, excel, Motif_S, do_info, info, test, motif_numb):
    for i in SL[deep - deeps][0]:
        try:
            repeat = info._get_value(i - 1, 'info')
            if repeat == 0 or do_info[1] == 0:
                continue
            else:
                if do_info[1] != repeat and repeat != 0:
                    motif_numb[str(12)] += 1
                else:
                    motif_numb[str(repeat) + str(repeat)] += 1
        except:
            cc = 1
        else:
            cc = 1


def Search_M33_2_1(MS, SL, deep, deeps, excel, Motif_S, do_info, info, test, motif_numb):
    for i in SL[deep - deeps][0]:
        try:
            repeat = info._get_value(i - 1, 'info')
            if repeat == 0 or do_info[1] == 0:
                continue
            else:
                motif_numb[str(do_info[1]) + str(repeat)] += 1
        except:
            cc = 1
        else:
            cc = 1
