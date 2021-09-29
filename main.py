import torch
import time

from models.net import SimpleNet as Net
from dataset import FeatureDataset, WenshuDataset


def main():
    model = Net()
    model.to('cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 4
    # dataset = WenshuDataset((0, 100))
    dataset = FeatureDataset(model, (0, 100))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(100):
        for x in dataloader:
            # print(x[0])
            n = len(x[0])
            # output0 = [model.encode(a) for a in x[0]]
            # output1 = [model.encode(a) for a in x[1]]
            output0 = [a for a in x[0]]
            output1 = [a for a in x[1]]

            positive, negative, loss = model.get_loss(output0, output1)
            print(positive.item(), negative.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    dataset = FeatureDataset(model, (10000, 10100))
    val = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    with torch.no_grad():
        for x in val:
            n = len(x[0])
            # output0 = [model.encode(a) for a in x[0]]
            # output1 = [model.encode(a) for a in x[1]]
            output0 = [a for a in x[0]]
            output1 = [a for a in x[1]]

            positive, negative, loss = model.get_loss(output0, output1)
            print(positive.item(), negative.item())

    
    


if __name__ == '__main__':
    startTime = time.process_time()
    main()
    print('the process time is: ', time.process_time() - startTime)
