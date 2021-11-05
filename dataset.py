import torch
import time
import random
from typing import List, Any, Tuple

class FeatureDataset(torch.utils.data.Dataset): # 好像没啥用了
    def __init__(self, features: List[str], part: tuple = None, relationPath='./datasets/pre/relation.pt') -> None:
        super().__init__()

        relation = torch.load(relationPath)
        if part is not None:
            relation = relation[part[0]:part[1]]

        features: List[List] = [torch.load(feature) for feature in features]
        print('FEATURES LOADED.')
        # print(features[0])

        need = {}
        for x in relation:
            for t in x:
                if t not in need:
                    need[t] = [feature[t] for feature in features]

        self.need = need
        self.relation = relation

    def __getitem__(self, index) -> Tuple[List, List]:
        x = self.relation[index]
        return self.need[x[0]], self.need[x[1]], (x[0],x[1])

    def __len__(self) -> int:
        return len(self.relation)

class PairDataset(torch.utils.data.Dataset): 
    def __init__(self, bertPath='./datasets/feature/bert.pt', part: tuple = None, relationPath='./datasets/splits/train.pt') -> None:
        super().__init__()

        relation = torch.load(relationPath)
        if part is not None:
            relation = relation[part[0]:part[1]]

        bert = torch.load(bertPath)
        print('Bert feature LOADED.')
        # print(features[0])

        self.bert = bert
        self.relation = relation

    def __getitem__(self, index) -> Tuple[List, List]:
        x = self.relation[index]
        return self.bert[x[0]], self.bert[x[1]], (x[0],x[1])

    def __len__(self) -> int:
        return len(self.relation)

class PNDataset(torch.utils.data.Dataset):
    def __init__(self, bertPath='./datasets/feature/bert.pt', relationPath='./datasets/pn/train.pt') -> None:
        super().__init__()

        relation = torch.load(relationPath)
        # print(relation[0])
        # exit()

        bert = torch.load(bertPath)
        print('Bert feature LOADED.')
        # print(features[0])

        self.bert = bert
        self.relation = relation

    def __getitem__(self, index) -> Tuple[List, List]:
        x = self.relation[index]
        a = x[0]
        b = random.sample(x[1],1)[0]
        if len(x[2]) == 0:
            c = random.sample(self.relation,1)[0][0]
        else:
            c = random.sample(x[2],1)[0]

        return self.bert[a], self.bert[b], self.bert[c], (a,b,c)

    def __len__(self) -> int:
        return len(self.relation)


class AllDataset:
    def __init__(self, bertPath='./datasets/feature/bert.pt', relationPath='./datasets/pn/train.pt', splitPath='./datasets/all/split.pt') -> None:
        relation = torch.load(relationPath)
        split = torch.load(splitPath)
        print('Relation LOADED.')
        # print(len(split))
        # print(len(relation))
        # print(split[0])
        # print(relation[0:10])
        # exit()

        bert = torch.load(bertPath)
        print('Bert feature LOADED.')

        self.bert = bert
        self.split = split
        self.relation = relation


class AllSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, part) -> None:
        super().__init__()
        self.dataset = dataset
        self.part = part
        data = [
            [x[0], x[1], [a for a in x[2] if self.dataset.split[a[0]] == self.part]]
            for x in self.dataset.relation
            if self.dataset.split[x[0]] == self.part
        ]
        self.data = [
            x for x in data
            if len(x[1]) > 0 or len(x[2]) > 0
        ]
        # print(self.dataset.split[self.dataset.relation[0][0]])
    
    def __getitem__(self, index):
        x = self.data[index]
        a = x[0]
        if (random.random() < 0.5 or len(x[2]) == 0) and len(x[1]) > 0:
            # if len(x[1]) == 0:
            #     print(x)
            b, num = random.sample(x[1],1)[0]
            flag = 0
        elif len(x[2]) > 0:
            b, num = random.sample(x[2],1)[0]
            flag = 1

        return self.dataset.bert[a], self.dataset.bert[b], flag, (a,b,num,flag)

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    startTime = time.process_time()
    # dataset = FeatureDataset([],part=(0, 100))
    # train = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/train.pt')
    # train = PNDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/pn/train.pt')
    dataset = AllDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/all/relation.pt', splitPath='/mnt/data/mzc/datasets/all/split.pt')
    train = AllSubset(dataset, 0)
    print(len(train))
    print(train[0])
    print('the process time is: ', time.process_time() - startTime)
