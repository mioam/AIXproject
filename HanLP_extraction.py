import hanlp
import torch
from tqdm import tqdm

def getNER(texts):
    ans = []
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    # out = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
    # print(out)
    # exit()
    for html_text in tqdm(texts):
        text = extract_plaintext(html_text)
        result = HanLP(text)['ner/msra']
        # print(result)
        ans.append(calc(result))

    return ans

from bs4 import BeautifulSoup
import jionlp as jio

def calc(x):
    
    LOC = {}
    ORG = {}
    PER = {}
    get = {'LOCATION':LOC, 'PERSON':PER, 'ORGANIZATION':ORG}
    for _ in x:
        for a in _:
            # print(a)
            if a[1] not in get:
                continue
            tmp = get[a[1]]
            if a[0] in tmp:
                tmp[a[0]] += 1
            else:
                tmp[a[0]] = 1
    return {'LOC': list(LOC.items()), 'ORG': list(ORG.items()), 'PER': list(PER.items()), }

def extract_plaintext(html_text):
    # 预处理！
    soup = BeautifulSoup(html_text, 'html.parser')
    # text = soup.get_text()
    text = []
    for i in soup.stripped_strings:
        tmp = ''.join(i.split())
        if tmp.startswith(('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员')):
            break
        tmp = jio.split_sentence(tmp, criterion='coarse')
        text.extend(tmp)
    # print(text)
    return text


def main():
    texts = torch.load('datasets/pre/content.pt')
    texts = getNER(texts[:10])
    # torch.save(texts,'datasets/feature/entity.pt')


if __name__ == "__main__":
    main()



