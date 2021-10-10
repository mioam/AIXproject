import torch
import time

from models.net import SimpleNet
from models.nef import NEFBasedNet

from dataset import FeatureDataset

num_epoch = 1

def main():
    model = NEFBasedNet()
    # model.to('cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 8
    dataset = FeatureDataset(['./datasets/feature/entity.pt'], (0, 100))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(num_epoch):
        for x in dataloader:
            n = len(x[0])
            positive = 0
            negative = 0
            for a in range(n):
                for b in range(n):
                    dis = model(x[0][a], x[1][b])
                    if a == b: 
                        positive += dis
                    else:
                        negative += dis

            print(positive.item(), negative.item())

            loss = positive - negative
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
 
    


if __name__ == '__main__':
    startTime = time.process_time()
    main()
    print('the process time is: ', time.process_time() - startTime)
