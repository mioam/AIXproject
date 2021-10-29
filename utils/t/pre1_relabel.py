import json
import glob
import re
import time
from tqdm import tqdm
from multiprocessing import Pool

import torch

def main():
    startTime = time.process_time()

    content = set()
    usedIds = set()
    relation = set()

    def add(x, y):
        # TODO: use UFS
        if x == y:
            return
        if (x in content) and (y in content):
            usedIds.add(x)
            usedIds.add(y)
            relation.add((x,y))
        # print(x,y)


    # contentList = glob.glob('./datasets/json/content100.json')
    contentList = glob.glob('./datasets/json/content[0-9]*.json')
    print('content list length is: ', len(contentList))
    for a in tqdm(contentList):
        with open(a, 'r', encoding='utf-8') as f:
            b = json.load(f)
            for e in b:
                content.add(e['id'])

    relList = json.load(open('./datasets/json/0.json', 'r', encoding='utf-8'))
    print('the relation list length is: ', len(relList))
    for e in tqdm(relList):
        x = e['id']
        for y in e['relWenshu']:
            add(x,y)

    print('the number of contents is: ', len(content))
    print('the number of used ids is: ', len(usedIds))
    print('the number of relations is: ', len(relation))

    contentNew = []
    labeledId = {}
    for a in tqdm(contentList):
        with open(a, 'r', encoding='utf-8') as f:
            b = json.load(f)
            for e in b:
                if e['id'] in usedIds:
                    usedIds.remove(e['id'])
                    labeledId[e['id']] = len(contentNew)
                    contentNew.append(e['content'])

    relationNew = []
    for x, y in relation:
        x = labeledId[x]
        y = labeledId[y]
        relationNew.append((x,y))

    torch.save(contentNew,'./datasets/pre/content.pt') # content (index)
    torch.save(relationNew,'./datasets/pre/relation.pt') # relation (index)
    torch.save(labeledId,'./datasets/pre/labeledId.pt') # dict docID -> index

    print('the process time is: ', time.process_time() - startTime)


if __name__ == '__main__':
    main()