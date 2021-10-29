import torch
import time
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
    pass


if __name__ == '__main__':
    startTime = time.process_time()
    # dataset = FeatureDataset([],part=(0, 100))
    train = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/train.pt')
    print('the process time is: ', time.process_time() - startTime)
