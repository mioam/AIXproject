import torch
import glob
import re
import time
from models.net import SimpleNet

from transformers import BertTokenizerFast, BertModel
from bs4 import BeautifulSoup

def func(args):
    tokenizer, e = args
    soup = BeautifulSoup(e, 'html.parser')
    text = soup.get_text()
    return text
    tokens = tokenizer('[CLS]' + text + '[SEP]',
                       add_special_tokens=False, return_tensors="pt")
    return tokens

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, model = SimpleNet(), part: tuple = None) -> None:
        super().__init__()
        self.model = model
        relation = torch.load('./datasets/pre/relation.pt')
        if part is not None:
            relation = relation[part[0]:part[1]]
        tokenizer = BertTokenizerFast.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')
        content = torch.load('./datasets/pre/content.pt')
        print('ok')
        need = {}
        for x in relation:
            for t in x:
                need[t] = self.func1((tokenizer, content[t]))

        self.need = need
        self.tokenizer = tokenizer
        self.relation = relation
        
    @torch.no_grad()
    def func1(self, args):
        text = func(args)
        feature = self.model.encode(text)
        return feature


    def __getitem__(self, index):
        x = self.relation[index]
        return self.need[x[0]], self.need[x[1]]

    def __len__(self) -> int:
        return len(self.relation)

class WenshuDataset(torch.utils.data.Dataset):
    def __init__(self, part: tuple = None) -> None:
        super().__init__()
        relation = torch.load('./datasets/pre/relation.pt')
        if part is not None:
            relation = relation[part[0]:part[1]]
        tokenizer = BertTokenizerFast.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')
        content = torch.load('./datasets/pre/content.pt')
        print('ok')
        need = {}
        for x in relation:
            for t in x:
                # need[t] = content[t]
                need[t] = func((tokenizer, content[t]))

        # need = {}
        # fileList = glob.glob('./datasets/pre/content_tokens/[0-9]*.pt')
        # maxn = 10000
        # requireList = [[] for i in range(100)]
        # for x in relation:
        #     requireList[x[0] // maxn].append(x[0])
        #     requireList[x[1] // maxn].append(x[1])
        # for fileName in fileList:
        #     start = re.search('([0-9]+)', fileName).group()
        #     start = int(start)
        #     print(fileName, len(requireList[start // maxn]))
        #     if len(requireList[start // maxn]) == 0:
        #         continue
        #     content = torch.load(fileName)
        #     for i in requireList[start // maxn]:
        #         need[i] = content[i - start]
        self.need = need
        self.tokenizer = tokenizer
        self.relation = relation

    def __getitem__(self, index):
        x = self.relation[index]
        return self.need[x[0]], self.need[x[1]]

    def __len__(self) -> int:
        return len(self.relation)


if __name__ == '__main__':
    startTime = time.process_time()
    dataset = WenshuDataset((0, 100))
    print('the process time is: ', time.process_time() - startTime)
