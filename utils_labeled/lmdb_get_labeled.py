import lmdb
from tqdm import tqdm
import json
import torch
from utils import build_map
import os

import matplotlib.pyplot as plt

def get_words(value):
    key_words = value[3]
    return [(word, Type)
    for Type, Words in key_words.items()
        for word, count in Words]

def getPER(key_words):
    return [word
        for word, count in key_words['PER']]

def compare(a, b): # naive version 比较相同的关键字
    x = set(getPER(a))
    y = set(getPER(b))
    z = x & y
    return len(z)

contents = torch.load('/mnt/data/mzc/datasets/pre/content.pt')
save_cnt = 0
def save(ty, a, b, candidate, important_words, rel):
    global save_cnt
    name = f'./_100/{ty}_{save_cnt}'
    os.makedirs(name)

    with open(name + f'/a_{a}.html','w') as f:
        f.write(contents[a])
        
    with open(name + f'/b_{b}.html','w') as f:
        f.write(contents[b])
    if ty == 'false':
        c = rel[0]
        with open(name + f'/c_{c}.html','w') as f:
            f.write(contents[c])

    with open(name + f'/info.txt','w') as f:
        f.write(str(candidate) + '\n')
        f.write(str(important_words) + '\n')
        f.write(str(rel) + '\n')
        f.write(str(json.loads(txn.get(str(a).encode()).decode())) + '\n')
        f.write(str(json.loads(txn.get(str(b).encode()).decode())) + '\n')
        if ty == 'false':
            f.write(str(json.loads(txn.get(str(c).encode()).decode())) + '\n')


    save_cnt += 1
    if save_cnt == 100:
        exit()

class UFS:
    def __init__(self, n):
        self.n = n
        self.fa = [i for i in range(n)]
    def getfa(self, x):
        if self.fa[x] != x:
            self.fa[x] = self.getfa(self.fa[x])
        return self.fa[x]
    def merge(self, x, y):
        fax = self.getfa(x)
        fay = self.getfa(y)
        self.fa[fax] = fay
    def count(self):
        a = [0 for i in range(self.n)]
        for i in range(self.n):
            a[self.getfa(i)] += 1
        a.sort()
        print([x for x in a if x > 0])
    def split(self, p):
        a = [0 for i in range(self.n)]
        for i in range(self.n):
            a[self.getfa(i)] += 1
        b = [i for i in range(self.n) if a[i] > 0]
        b.sort(key=lambda x: -a[x])

        weights = torch.tensor([self.n * i for i in p])
        ret = [0 for i in range(self.n)]
        for i in b:
            x = torch.multinomial(torch.nn.functional.relu(weights) , 1)
            ret[i] = x
            weights[x] -= a[i]

        print(weights)
        return ret



if __name__ =='__main__':

    env = lmdb.open('/mnt/data/mzc/key_tmp')
    env_map = lmdb.open('/mnt/data/mzc/map_tmp')

    txn = env.begin(write=False)
    txn_map = env_map.begin(write=False)
    print(txn.stat())
    print(txn_map.stat())

    relation = torch.load('/mnt/data/mzc/datasets/pre/relation.pt')
    rel = {}
    rel_pair = {}
    for x, y in relation:
        rel_pair[(x,y)] = True
        rel_pair[(y,x)] = True
        if x not in rel:
            rel[x] = []
        if y not in rel:
            rel[y] = []
        rel[x].append(y)
        rel[y].append(x)
    for x in rel.keys():
        rel[x] = list(set(rel[x]))# 去重

    # print(rel[100065])
    # print(rel[90028])
    # print(rel[169439])
    # exit()

    # ufs = UFS(len(rel))
    # for x, y in relation:
    #     ufs.merge(x,y)
    # split = ufs.split([0.7,0.2,0.1])

    # DATASET = [[],[],[]]
    # for x, y in relation:
    #     DATASET[split[x]].append((x,y))
    # print([len(x) for x in DATASET])
    # torch.save(DATASET[0],'/mnt/data/mzc/datasets/splits/train.pt')
    # torch.save(DATASET[1],'/mnt/data/mzc/datasets/splits/valid.pt')
    # torch.save(DATASET[2],'/mnt/data/mzc/datasets/splits/test.pt')


    # MAX = max([len(set(r)) for r in rel.values()])
    # print(MAX)

    # exit()

    pn_relation = []

    truepossitive = []
    falsepossitive = []
    length = []
    print(len(rel_pair))

    
    WEIGHT = {'LOC':1, 'ORG':1, 'PER':0}

    # key = '100065'
    # key = '208140'
    # key = '123456'
    # value = txn.get(key.encode()).decode()
    # if True:
    for key, value in tqdm(txn.cursor()):
        value = json.loads(value)
        # print(key)
        key_words = value[3]
        candidate = {}
        important_words = []
        ma = 0
        for word, Type in get_words(value):
            if txn_map.get(build_map.map_key(word,build_map.Threshold-1)) is not None:
                continue
            ma += 1
            important_words.append((word, Type))
            # print(word, Type)
            # exit()
            for i in range(build_map.Threshold):
                ret = txn_map.get(build_map.map_key(word,i))
                if ret is None:
                    break
                else:
                    if ret not in candidate:
                        b = json.loads(txn.get(ret).decode())[3]
                        a = value[3]
                        candidate[ret] = compare(a,b) * 3 + WEIGHT[Type]
                    else:
                        candidate[ret] += WEIGHT[Type]
                    # if ret in candidate:
                    #     candidate[ret] += WEIGHT[Type]
                    # else:
                    #     candidate[ret] = WEIGHT[Type]
        candidate = list(candidate.items())
        candidate.sort(key=lambda x:-x[1])
        # print(candidate)
        # if len(candidate) > 4000:
        #     print(key)
        #     exit()
        length.append(len(candidate))
        # print(rel[int(key)])
        x = int(key)

        now = [x, rel[x], [] ]

        for i, _ in enumerate(candidate):
            y, num = _
            y = int(y)
            if x == y:
                continue
            if (x,y) in rel_pair:
                truepossitive.append(i)
                # save('true', x, y, candidate, important_words, rel[int(key)])
            else:
                falsepossitive.append(i)
                now[2].append(y)
                if len(now[2]) == max(2, len(now[1])):
                    break
            #     save('false', x, y, candidate, important_words, rel[int(key)])
            # break
        pn_relation.append(now)

        # break

    
    ufs = UFS(len(rel))
    for x in pn_relation:
        for y in x[1]+x[2]:
            ufs.merge(x[0],y)
            
    split = ufs.split([0.7,0.2,0.1])

    DATASET = [[],[],[]]
    for x in pn_relation:
        DATASET[split[x[0]]].append(x)
    print([len(x) for x in DATASET])
    torch.save(DATASET[0],'/mnt/data/mzc/datasets/pn/train.pt')
    torch.save(DATASET[1],'/mnt/data/mzc/datasets/pn/valid.pt')
    torch.save(DATASET[2],'/mnt/data/mzc/datasets/pn/test.pt')

    env.close()
    env_map.close()
    
    # print(len(truepossitive), len(rel_pair))

    # plt.hist(falsepossitive,bins=range(1,100),log=False,alpha=0.8, label ='falsepossitive')
    # plt.hist(truepossitive,bins=range(1,100),log=False,alpha=0.8, label ='truepossitive')
    # plt.legend()
    # plt.savefig(f'positive.png')
    # plt.cla()

    # plt.hist(length,log=True,alpha=0.8, label ='length')
    # plt.legend()
    # plt.savefig(f'length.png')
    # plt.cla()