import torch
from torch.nn import functional as F
import time

from models.net import SimpleNet
from models.nef import NEFBasedNet
from models.bert import BertNet

from dataset import FeatureDataset

num_epoch = 1

def main():
    model = BertNet()
    # model = NEFBasedNet()
    # model.to('cuda')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)

    batch_size = 8
    # dataset = FeatureDataset(['./datasets/feature/entity.pt'], (0, 100))
    dataset = FeatureDataset(['./datasets/feature/bert.pt'])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(num_epoch):
        for x, y, ex in dataloader:
            n = len(x[0])
            # print(n)
            positive = []
            negative = []
            for a in range(n):
                for b in range(n):
                    dis = model(x[0][a], y[0][b])
                    if a == b: 
                        positive.append(dis)
                    else:
                        negative.append(dis)
            positive = torch.stack(positive)
            negative = torch.stack(negative)

            print(positive[0].item(), negative[0].item(), (ex[0][0], ex[1][0]), (ex[0][0], ex[1][1]))

            loss = F.relu(0.5-positive).sum() + F.relu(0.5+negative).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
 
    


if __name__ == '__main__':
    startTime = time.process_time()
    main()
    print('the process time is: ', time.process_time() - startTime)
