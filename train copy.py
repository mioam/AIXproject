import torch
from torch.nn import functional as F
import time
import os
import numpy as np
import random

from models.net import SimpleNet
from models.ner import NEFBasedNet
from models.bert import BertNet

from dataset import FeatureDataset, PairDataset, PNDataset

import argparse
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--num-epoch", default=100, type=int)
    parser.add_argument("--optim", default="AdamW", choices=["Adam", "AdamW"])
    parser.add_argument("--Type", default="PN", choices=["pair", "PN", "PNplus"])

    parser.add_argument("--load-dir", default='')
    args = parser.parse_args()
    return args

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def eva(model, loss_fn, dataloader, step, task_name=None):
    global device
    model.eval()
    tot_loss = 0
    tot_true = 0
    tot = 0
    tot_l = 0
    for sample in dataloader:
        if args.Type == 'pair': 
            x, y, ex = sample
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
            tot_true += (dis.argmax(1) == torch.arange(n).to(device)).sum()
        elif args.Type == 'PN':
            anchor, positive, negative, ex = sample
            n = anchor.shape[0]
            _anchor = model(anchor.to(device))
            _positive = model(positive.to(device))
            _negative = model(negative.to(device))
            pos = (_anchor * _positive).sum(1)
            neg = (_anchor * _negative).sum(1)
            loss = loss_fn(pos, neg)
            tot += n
            tot_l += 1
            tot_loss += loss.item()
            tot_true += (pos > neg).sum()
        elif args.Type == 'PNplus':
            anchor, positive, negative, ex = sample
            n = anchor.shape[0]
            _anchor = model(anchor.to(device))
            _positive = model(positive.to(device))
            _negative = model(negative.to(device))
            
            X = torch.cat([_anchor, _positive, _negative], 0)
            dis = torch.mm(_anchor, X.permute(1,0))
            mask = torch.eye(n, 3*n).to(device)
            dis = dis - mask * 10000000

            loss = loss_fn(dis)
            tot += n
            tot_l += 1
            tot_loss += loss.item()
            tot_true += (dis.argmax(1) == (n+torch.arange(n)).to(device)).sum()
        elif 


    print(f'step: {step}, tot: {tot}, tot_loss: {tot_loss}, avg_loss: {tot_loss / tot_l}, tot_true: {tot_true}, acc: {tot_true / tot}')

    if task_name is not None:
        global writer
        writer.add_scalar(f'loss/{task_name}', tot_loss / tot_l, step)
        writer.add_scalar(f'acc/{task_name}', tot_true / tot, step)

    return tot_true / tot


def naive_loss(dis): # 真不行
    device = dis.device
    n = dis.shape[0]

    weight = torch.eye(n) * 2 - 1
    weight = weight.to(device)

    loss = (dis * weight).sum()
    return -loss

def simple_loss(dis): # 还可以
    device = dis.device
    n = dis.shape[0]

    weight = torch.eye(n)
    weight = weight.to(device)

    loss = (dis * weight).sum() - ((dis - weight * 1000000).max(1).values).sum()
    return -loss

def pn_loss(pos, neg): # 不行 0.6
    loss = pos - neg
    return -loss.sum() / pos.shape[0]

def pnplus_loss(dis):
    device = dis.device
    n = dis.shape[0]
    m = dis.shape[1]

    diag = torch.cat((torch.zeros((n,n)),torch.eye(n),torch.zeros((n,n)),),1)
    diag = diag.to(device)

    loss = ((dis * diag).sum() - ((dis - diag * 1000000).max(1).values).sum()) / n
    return -loss

def main():
    
    global args
    global writer
    global device
    set_random(args.seed)

    model = BertNet()
    model.to(device)
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise 'Unknown optim'

    # dataset = FeatureDataset(['./datasets/feature/entity.pt'], (0, 100))
    # dataset = FeatureDataset(['./datasets/feature/bert.pt'])
    if args.Type == 'pair':
        train_dataset = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/train.pt')
        valid_dataset = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/valid.pt')
        test_dataset = PairDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/splits/test.pt')
    elif args.Type == 'PN' or args.Type == 'PNplus':
        train_dataset = PNDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/pn/train.pt')
        valid_dataset = PNDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/pn/valid.pt')
        test_dataset = PNDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/pn/test.pt')
    else:
        raise 'Unknown Type'

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    
    if args.Type == 'pair': 
        # loss_fn = naive_loss
        loss_fn = simple_loss
    elif args.Type == 'PN':
        loss_fn = pn_loss
    elif args.Type == 'PNplus':
        loss_fn = pnplus_loss

    step = 0
    eva(model, loss_fn, valid_dataloader, step, task_name='valid')
    for epoch in range(args.num_epoch):
        model.train()
        for sample in train_dataloader:

            if args.Type == 'pair': 
                x, y, ex = sample
                n = x.shape[0]
                x = x.to(device)
                y = y.to(device)
                _x = model(x)
                _y = model(y)
                dis = torch.mm(_x, _y.permute(1,0))
                loss = loss_fn(dis)
            elif args.Type == 'PN':
                anchor, positive, negative, ex = sample
                _anchor = model(anchor.to(device))
                _positive = model(positive.to(device))
                _negative = model(negative.to(device))
                pos = (_anchor * _positive).sum(1)
                neg = (_anchor * _negative).sum(1)
                loss = loss_fn(pos, neg)
            elif args.Type == 'PNplus':
                anchor, positive, negative, ex = sample
                n = anchor.shape[0]
                _anchor = model(anchor.to(device))
                _positive = model(positive.to(device))
                _negative = model(negative.to(device))
                
                X = torch.cat([_anchor, _positive, _negative], 0)
                dis = torch.mm(_anchor, X.permute(1,0))
                mask = torch.eye(n, 3*n).to(device)
                dis = dis - mask * 10000000

                loss = loss_fn(dis)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            
            writer.add_scalar('loss/train', loss.item(), step)
            if step % 1000 == 0 :
                eva(model, loss_fn, train_dataloader, step, task_name='train2')
                acc = eva(model, loss_fn, valid_dataloader, step, task_name='valid')
                global BEST
                global NAME
                if (BEST is None) or (acc > BEST):
                    BEST = acc
                    torch.save({'model': model.state_dict(), 'step': step, 'acc': acc, 'NAME': NAME}, os.path.join('./checkpoints', NAME + '.pt'))

                model.train()

    
    path = os.path.join('./checkpoints', NAME +
                        '.pt') if args.load_dir == '' else args.load_dir
    model.load_state_dict(torch.load(path)['model'])

    eva(model, loss_fn, test_dataloader, step, task_name='test')



if __name__ == '__main__':
    startTime = time.process_time()
    
    device = 'cuda'

    ti = time.time()
    args = get_args()
    BEST = None
    NAME = f'{args.seed}_{args.optim}_{args.lr}_{args.batch}_{args.weight_decay}_{args.Type}'

    TIME = f'_time{ti}'
    print(NAME, TIME)
    writer = SummaryWriter(log_dir=os.path.join('runs', NAME+TIME))
    main()
    print('the process time is: ', time.process_time() - startTime)
