import torch
import time

from dataset import FeatureDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

ty2key= ['LOC', 'ORG', 'PER']
def comp(a, b):
    ans = [0,0,0]
    for ty in range(3):
        for x in a[ty2key[ty]]:
            for y in b[ty2key[ty]]:
                if x == y:
                    ans[ty] += 1
    return ans

def main():

    batch_size = 16
    dataset = FeatureDataset(['./datasets/feature/entity.pt'])
    # dataset = FeatureDataset(['./datasets/feature/bert.pt'])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda x:(
            [a[0] for a in x], [a[1] for a in x], [a[2] for a in x]
        ))

    positive = [[],[],[]]
    negative = [[],[],[]]
    for x, y, ex in tqdm(dataloader):
        n = len(ex)
        # print(n)
        # print(x)
        # print(y)
        # print(ex)
        for a in range(n):
            for b in range(n):
                dis = comp(x[a][0], y[b][0])
                if a == b: 
                    for ty in range(3):
                        positive[ty].append(dis[ty])
                    
                else:
                    for ty in range(3):
                        negative[ty].append(dis[ty])

    for ty in range(3):
        plt.hist(positive[ty],bins=range(0,30),label='pos',alpha=.7,density=True)
        plt.hist(negative[ty],bins=range(0,30),label='neg',alpha=.7,density=True)
        plt.legend()
        plt.savefig(f'{ty2key[ty]}.png')
        plt.cla()
    # print(positive[0].item(), negative[0].item())

 

if __name__ == '__main__':
    startTime = time.process_time()
    main()
    print('the process time is: ', time.process_time() - startTime)
