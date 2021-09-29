import json
import glob
import re
import time
from tqdm import tqdm
from multiprocessing import Pool

import torch
from transformers import BertTokenizerFast, BertModel


def func(args):
    tokenizer, e = args
    text = re.sub('(<.*?>)+','[SEP]',e['content'])
    text = re.sub('^\[SEP\]','',text)
    tokens = tokenizer('[CLS]' + text, add_special_tokens=False, return_tensors="pt")
    return e['id'], tokens

def main():
    startTime = time.process_time()

    content = {} # content[docId] = tokens
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

    tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

    # contentList = glob.glob('./datasets/json/content100.json')
    contentList = glob.glob('./datasets/json/content[0-9]*.json')
    print('content list length is: ', len(contentList))
    for a in tqdm(contentList):
        with open(a, 'r', encoding='utf-8') as f:
            b = json.load(f)
            # with Pool(3) as p:
            #     result = list(tqdm(p.imap(func, [(tokenizer, e) for e in b]), total=len(b), desc=a))
            #     for id, tokens in result:
            #         content[id] = tokens
            result = list(tqdm(map(func, [(tokenizer, e) for e in b]), total=len(b), desc=a))
            for id, tokens in result:
                content[id] = tokens

    relList = json.load(open('./datasets/json/0.json', 'r', encoding='utf-8'))
    print('the relation list length is: ', len(relList))
    for e in tqdm(relList):
        x = e['id']
        for y in e['relWenshu']:
            add(x,y)

    print('the number of contents is: ', len(content))
    print('the number of used ids is: ', len(usedIds))
    print('the number of relations is: ', len(relation))

    # tmp = list(content.keys())
    for key in list(content.keys()):
        if key not in usedIds:
            content.pop(key)

    print('the number of used contents is: ', len(content))

    contentRelabeled = []
    labeledId = {}
    relationNew = []
    for x, y in relation:
        def relabel(a):
            if x not in labeledId:
                labeledId[a] = len(contentRelabeled)
                contentRelabeled.append(content[a])
            return labeledId[a]
        x = relabel(x)
        y = relabel(y)
        relationNew.append((x,y))

    torch.save(contentRelabeled,'./datasets/pre/content.pt')
    torch.save(relationNew,'./datasets/pre/relation.pt')
    torch.save(labeledId,'./datasets/pre/labeledId.pt')

    print('the process time is: ', time.process_time() - startTime)


if __name__ == '__main__':
    main()