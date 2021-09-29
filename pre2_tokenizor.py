import glob
import re
import time
from tqdm import tqdm
from multiprocessing import Pool

import torch
from transformers import BertTokenizerFast, BertModel
from bs4 import BeautifulSoup


def func(args):
    tokenizer, e = args
    soup = BeautifulSoup(e, 'html.parser')
    text = soup.get_text()
    tokens = tokenizer('[CLS]' + text + '[SEP]',
                       add_special_tokens=False, return_tensors="pt")
    return tokens


def main():
    startTime = time.process_time()

    tokenizer = BertTokenizerFast.from_pretrained(
        'hfl/chinese-roberta-wwm-ext')
    # bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

    content = torch.load('./datasets/pre/content.pt')

    # with Pool(3) as p:
    #     result = list(tqdm(p.imap(func, [(tokenizer, e) for e in b]), total=len(b), desc=a))
    #     for id, tokens in result:
    #         content[id] = tokens
    print(len(content))
    n = len(content)
    maxn = 10000
    for i in range(0, n, maxn):
        content_now = content[i:i+maxn]
        content_now = list(tqdm(map(func, [(tokenizer, e) for e in content_now]), total=len(content_now)))
        torch.save(content_now, f'./datasets/pre/content_tokens/{i}.pt')

    print('the process time is: ', time.process_time() - startTime)


if __name__ == '__main__':
    main()
