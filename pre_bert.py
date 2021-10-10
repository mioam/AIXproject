import time
from tqdm import tqdm
from multiprocessing import Pool

import torch
from transformers import BertTokenizerFast, BertModel
from bs4 import BeautifulSoup

DEVICE = 'cuda'
STEP = 400
WINDOW = 500

@torch.no_grad()
def func(args):
    tokenizer, bert, e = args

    soup = BeautifulSoup(e, 'html.parser')
    text = soup.get_text()

    outputs = []
    for i in range(0, len(text), STEP):
        s = text[i:i+WINDOW]
        tokens = tokenizer(s, return_tensors="pt", padding=True).to(DEVICE)
        outputs.append(bert(**tokens).pooler_output[0].cpu())
    outputs = torch.mean(torch.stack(outputs, dim=0), dim=0)  # mean
    return outputs


def main():
    startTime = time.process_time()

    tokenizer = BertTokenizerFast.from_pretrained(
        'hfl/chinese-roberta-wwm-ext')
    bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    bert.to(DEVICE)

    content = torch.load('./datasets/pre/content.pt')
    content = content[:10000]

    n = len(content)
    feature = list(tqdm(map(func, [(tokenizer, bert, e) for e in content]), total=n))
    torch.save(feature, f'./datasets/feature/bert.pt')

    print('the process time is: ', time.process_time() - startTime)


if __name__ == '__main__':
    main()
