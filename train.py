import torch
from torch.nn import functional as F
import time

from models.net import SimpleNet
from models.ner import NEFBasedNet
from models.bert import BertNet

from dataset import FeatureDataset, PairDataset


num_epoch = 100

@torch.no_grad()
def eval(model, loss_fn, dataloader, device, step):
    model.eval()
    tot_loss = 0
    tot_true = 0
    tot = 0
    tot_l = 0
    for x, y, ex in dataloader:
        n = x.shape[0]
        
        x = x.to(device)
        y = y.to(device)

        _x = model(x)
        _y = model(y)
        dis = torch.mm(_x, _y.permute(1,0))
        loss = loss_fn(dis)

        tot += n
        tot_l += 1
        tot_loss += loss.item()
        # print(dis.shape, dis.argmax(1).shape, torch.arange(n).shape)
        tot_true += (dis.argmax(1) == torch.arange(n).to(device)).sum()
    print(f'step: {step}, tot: {tot}, tot_loss: {tot_loss}, avg_loss: {tot_loss / tot_l}, tot_true: {tot_true}, acc: {tot_true / tot}')


def naive_loss(dis):
    device = dis.device
    n = dis.shape[0]

    weight = torch.eye(n) * 2 - 1
    weight = weight.to(device)

    loss = (dis * weight).sum()
    return -loss

def simple_loss(dis):
    device = dis.device
    n = dis.shape[0]

    weight = torch.eye(n)
    weight = weight.to(device)

    loss = (dis * weight).sum() - ((dis - weight * 1000000).max(1).values).sum()
    return -loss

def main():
    device = 'cuda'
    model = BertNet()
    # model = NEFBasedNet()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    batch_size = 512
    # dataset = FeatureDataset(['./datasets/feature/entity.pt'], (0, 100))
    # dataset = FeatureDataset(['./datasets/feature/bert.pt'])
    train_dataset = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/train.pt')
    valid_dataset = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/valid.pt')


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # loss_fn = naive_loss
    loss_fn = simple_loss

    step = 0
    eval(model, loss_fn, valid_dataloader, device, step)
    for epoch in range(num_epoch):
        model.train()
        for x, y, ex in train_dataloader:

            n = x.shape[0]
            
            x = x.to(device)
            y = y.to(device)

            _x = model(x)
            _y = model(y)
            dis = torch.mm(_x, _y.permute(1,0))
            loss = loss_fn(dis)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10000 == 0 :
                eval(model, loss_fn, valid_dataloader, device, step)
                model.train()

    


if __name__ == '__main__':
    startTime = time.process_time()
    main()
    print('the process time is: ', time.process_time() - startTime)
