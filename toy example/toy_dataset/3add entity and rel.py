#  @author rzh
#  @create 2023-12-20 9:21
import json
import pandas as pd

relation2ids = json.load(open('relation2ids'))
ent2ids = json.load(open('ent2ids'))

df = pd.read_csv('dti_rel.csv', sep=",", header=None)
rel = df.values.tolist()
df = pd.read_csv('dti_entity.csv', sep=",", header=None)
entity = df.values.tolist()

index = len(relation2ids)
for i in rel:
    if i[0] not in relation2ids:
        relation2ids[i[0]] = index
        index+=1

index = len(ent2ids)
for i in entity:
    if i[0] not in ent2ids:
        ent2ids[i[0]] = index
        index+=1

json.dump(relation2ids, open('relation2ids', 'w'))
json.dump(ent2ids, open('ent2ids', 'w'))
