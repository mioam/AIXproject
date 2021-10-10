import torch
import time
from typing import List, Any, Tuple



class FeatureDataset(torch.utils.data.Dataset): 
    def __init__(self, features: List[str], part: tuple = None, relationPath='./datasets/pre/relation.pt') -> None:
        super().__init__()

        relation = torch.load(relationPath)
        if part is not None:
            relation = relation[part[0]:part[1]]

        features: List[List] = [torch.load(feature) for feature in features]
        print('FEATURES LOADED.')

        need = {}
        for x in relation:
            for t in x:
                if t not in need:
                    need[t] = [feature[t] for feature in features]

        self.need = need
        self.relation = relation

    def __getitem__(self, index) -> Tuple[List, List]:
        x = self.relation[index]
        return self.need[x[0]], self.need[x[1]]

    def __len__(self) -> int:
        return len(self.relation)


if __name__ == '__main__':
    startTime = time.process_time()
    dataset = FeatureDataset([],part=(0, 100))
    print('the process time is: ', time.process_time() - startTime)
