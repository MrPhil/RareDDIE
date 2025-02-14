from numpy import *
import numpy as np
import random
import math
import os
import time
import pandas as pd
import csv
import math
import random
import copy
import json

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StoreFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter='\t')
        writer.writerows(data)
    return
def StoreFile2(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerows(data)
    return

# construct ent2ids，path_graph，relation2ids   construct task
druglist = []
ReadMyCsv(druglist,'druglist.csv')
print(len(druglist))
drugdict = {}
for i in range(len(druglist)):
    drugdict[druglist[i][0]]=i
json.dump(drugdict, open('./ent2ids', 'w'))

data = pd.read_csv('toy.data', delimiter='\t')
a=[]
b={}
task={}
for id1, id2, smiles1, smiles2, relation, map in zip(data['ID1'], data['ID2'],data['X1'],data['X2'],data['Y'],data['Map'],):
    a.append([id1,map,id2])
    b[map]=relation-1
    if map not in task.keys():
        task[map]=[[id1,map,id2]]
    else:
        task[map].append([id1,map,id2])
StoreFile(a,'./path_graph')
json.dump(b, open('./relation2ids', 'w'))
json.dump(task, open('./alltask', 'w'))
data[['ID1','ID2','Y']].to_csv('pos_triplets.csv', index=False, header=False)

# distribution
aaa={}
for i in list(task.keys()):
    aaa[b[i]]=len(task[i])
bbb=sorted(aaa.items(),key = lambda x:x[1],reverse = True)
c=sorted(list(aaa.values()),reverse = True)
for num in range(len(c)):
    if c[num] < 50:
        com_num=num
        break
for num in range(len(c)):
    if c[num] < 20:
        few_num = num
        break

# 构建三类事件
Common_events = {}
Fewer_events = {}
Rare_events = {}
csv_reader = bbb
i=0
for row in csv_reader:
    if i <com_num:
        Common_events[int(row[0])] = int(row[1])
        i+=1
    elif i<few_num:
        Fewer_events[int(row[0])] = int(row[1])
        i += 1
    else:
        Rare_events[int(row[0])] = int(row[1])
        i += 1

def random_select(lst, num=5):
    return random.sample(lst, num) if len(lst) >= num else lst
selected_items = random_select(list(Common_events.keys()))

train_tasks ={}
dev_tasks ={}
test_tasks ={}
test2_tasks ={}
for t in task.keys():
    if b[t] in Fewer_events.keys():
        test_tasks[t] = task[t]
    if b[t] in Rare_events.keys():
        test2_tasks[t] = task[t]
    if b[t] in Common_events.keys() and b[t] in selected_items:
        dev_tasks[t] = task[t]
    if b[t] in Common_events.keys() and b[t] not in selected_items:
        train_tasks[t] = task[t]
json.dump(train_tasks, open('./train_tasks.json', 'w'))
json.dump(dev_tasks, open('./dev_tasks.json', 'w'))
json.dump(test_tasks, open('./test_tasks.json', 'w'))
json.dump(test2_tasks, open('./test2_tasks.json', 'w'))
