import torch
from torch import nn
from torch.nn import functional as F
import time
import os
import numpy as np
import random

from models.classifier import Net, AttnNet

from dataset import FeatureDataset, PairDataset, PNDataset, AllDataset, AllSubset

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
    parser.add_argument("--Type", default="default", choices=["default",])


    parser.add_argument("--netType", default='Net', choices=["Net", "Attn"])
    parser.add_argument("--num-layer", default=2, type=int)
    parser.add_argument("--act", default="ReLU", choices=["ReLU", "GLU"])
    parser.add_argument("--hidden-size", default=768, type=int)
    parser.add_argument("--feature-type", default=0, type=int)

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

    true_pos = 0
    true_neg = 0
    pos = 0
    neg = 0

    tot = 0
    tot_l = 0
    hist = []
    for sample in dataloader:
        if args.Type == 'default': 
            x, y, label, ex = sample
            # print(ex)
            # exit()
            n = x.shape[0]
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            output = model(x,y)
            # print(output.shape)
            # print(label.shape, label)
            loss = loss_fn(output, label)
            ans = output.argmax(1)
            true_pos += torch.logical_and(ans == 0, label == 0).sum()
            true_neg += torch.logical_and(ans == 1, label == 1).sum()
            pos += (label == 0).sum()
            neg += (label == 1).sum()
            for i in range(n):
                hist.append((ex[2][i].item(), ans[i].item(), label[i].item()))
                # if ex[2][i] != -1:
                #     if ans[i] == label[i]:
                #         hist_true.append(ex[2][i])
                #     else:
                #         hist_false.append(ex[2][i])
        tot += n
        tot_l += 1
        tot_loss += loss

    print(f'step: {step}, pos: {pos}, neg: {neg}, avg_loss: {tot_loss / tot_l}, true_pos acc: {true_pos / pos}, true_neg acc: {true_neg / neg}')

    if task_name is not None:
        global writer
        writer.add_scalar(f'loss/{task_name}', tot_loss / tot_l, step)
        writer.add_scalar(f'true_pos acc/{task_name}', true_pos / pos, step)
        writer.add_scalar(f'true_neg acc/{task_name}', true_neg / neg, step)
    if task_name == 'best':
        global NAME
        if not os.path.exists('./hist'):
            os.makedirs('./hist') 
        torch.save(hist, f'./hist/{NAME}.pt')

    return (true_pos + true_neg) / tot


def main():
    
    global args
    global writer
    global device
    set_random(args.seed)

    if args.netType == 'Net':
        model = Net(num_layer=args.num_layer,act=args.act,hidden_size=args.hidden_size,feature_type=args.feature_type)
    else:
        model = AttnNet(96, 8 ,4)
    model.to(device)

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise 'Unknown optim'

    # dataset = FeatureDataset(['./datasets/feature/entity.pt'], (0, 100))
    # dataset = FeatureDataset(['./datasets/feature/bert.pt'])
    if args.Type == 'default':
        dataset = AllDataset(bertPath='/mnt/data/mzc/datasets/feature/bert.pt', relationPath='/mnt/data/mzc/datasets/all/relation.pt', splitPath='/mnt/data/mzc/datasets/all/split.pt')
        train_dataset = AllSubset(dataset, 0)
        valid_dataset = AllSubset(dataset, 1, rd=False)
    else:
        raise 'Unknown Type'

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    
    if args.Type == 'default':
        loss_fn = nn.CrossEntropyLoss()

    step = 0
    # eva(model, loss_fn, valid_dataloader, step, task_name='valid')
    for epoch in range(args.num_epoch):
        model.train()
        for sample in train_dataloader:

            if args.Type == 'default': 
                x, y, label, ex = sample
                n = x.shape[0]
                x = x.to(device)
                y = y.to(device)
                label = label.to(device)
                output = model(x,y)
                loss = loss_fn(output, label)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            
            writer.add_scalar('loss/train', loss.item(), step)
            if step % 10000 == 0 :
                # eva(model, loss_fn, train_dataloader, step, task_name='train2')
                acc = eva(model, loss_fn, valid_dataloader, step, task_name='valid')
                global BEST
                global NAME
                if (BEST is None) or (acc > BEST):
                    BEST = acc
                    if not os.path.exists('./checkpoints'):
                        os.makedirs('./checkpoints') 
                    torch.save({'model': model.state_dict(), 'step': step, 'acc': acc, 'NAME': NAME}, os.path.join('./checkpoints', NAME + '.pt'))

                model.train()

    
    path = os.path.join('./checkpoints', NAME +
                        '.pt') if args.load_dir == '' else args.load_dir
    model.load_state_dict(torch.load(path)['model'])

    eva(model, loss_fn, valid_dataloader, step, task_name='best')



if __name__ == '__main__':
    startTime = time.process_time()
    
    device = 'cuda'

    ti = time.time()
    args = get_args()
    BEST = None
    NAME = f'{args.seed}_{args.optim}_{args.lr}_{args.batch}_{args.weight_decay}_{args.Type}_{args.netType}'
    if args.netType == 'Net':
        NAME += f'_{args.num_layer}_{args.act}_{args.hidden_size}_{args.feature_type}'
    else:
        NAME += ''
    TIME = f'_time{ti}'
    print(NAME, TIME)
    writer = SummaryWriter(log_dir=os.path.join('runs', NAME+TIME))
    main()
    print('the process time is: ', time.process_time() - startTime)
