import json
import glob
import re
import time
from tqdm import tqdm
import csv
import os
import lmdb
import hanlp

import torch
from bs4 import BeautifulSoup

def magic(tmp):

    # # 删除乱码
    # if row[2].find('"s1":!') != -1:
    #     return None
    # if row[2].find('"s1":#') != -1:
    #     return None
    tmp = str(tmp)
    tmp = tmp[1:-1]
    tmp = tmp.replace('""','"')
    tmp = tmp.replace('\n','')
    tmp = tmp.replace('\\','\\\\')
    x = tmp.find('"qwContent":"')
    if x == -1: # 没有文本
        return None


    # 转义引号
    x += 13 # "qwContent":"
    y = tmp.find('","directory"',x)
    if y == -1:
        y = tmp.find('","viewCount"',x)
    tmp = tmp[:x] + tmp[x:y].replace(r'"',r'\"') + tmp[y:]

    # # 相关文书
    # # if tmp.find('"relWenshu"') == -1:
    # #     return None
    # # x0 = tmp.find('"relWenshu"')
    # # y0 = tmp.find(']',x0)+2
    # # if y0 - x0 != 15:
    # #     cnt += 1
    
    j = json.loads(tmp)
    
    # try:
    #     j = json.loads(tmp)
    # except BaseException:
    #     print('ERROR')
        
    #     # print(x,y)
    #     with open('orign', 'w', encoding='UTF-8') as f:
    #         f.write(tmp)
    #     # with open('after', 'w', encoding='UTF-8') as f:
    #     #     f.write(eval(tmp))
    #     # exit()
    
    if 'DocInfoVo' in j: # unknown format
        return None
    
    # with open('orign', 'w', encoding='UTF-8') as f:
    #     f.write(tmp)
    if j['qwContent'] == '':
        return None
    # print(row[0], row[2])
    return j['qwContent']


MAX_LENGTH = 200

def split_sentence(text):
    sentence = hanlp.utils.rules.split_sentence(text)
    ret = []
    for s in sentence:
        if len(s) > MAX_LENGTH:
            sr = re.sub('([,;，；])', r"\1\n", s)
            sr = sr.split('\n')
            l = MAX_LENGTH
            for x in sr:
                x = x.strip()
                if x == '':
                    continue
                if len(x) > MAX_LENGTH:
                    for t in range(0,len(x),MAX_LENGTH):
                        ret.append(x[t:t+MAX_LENGTH])
                    l = len(ret[-1])
                elif len(x) + l > MAX_LENGTH:
                    ret.append(x)
                    l = len(ret[-1])
                else:
                    ret[-1] = ret[-1] + x
                    l = len(ret[-1])
        else:
            ret.append(s)
    return ret

def extract_plaintext(html_text):
    # 预处理！
    soup = BeautifulSoup(html_text, 'html.parser')
    # text = soup.get_text()
    text = []
    for i in soup.stripped_strings:
        tmp = ''.join(i.split())
        if tmp.startswith(('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员')):
            break
        # tmp = jio.split_sentence(tmp, criterion='coarse')
        # print(tmp)
        text.append(tmp)
    # print(text)
    text = list(split_sentence('\n'.join(text)))
    return text

def get_key(HanLP, plain_text_arr):
    # print(plain_text_arr)
    if plain_text_arr == []:
        return []
    result = HanLP(plain_text_arr)
    result = result['ner']
    return result
    # sort_arr = list(enumerate(plain_text_arr))
    # sort_arr.sort(key=lambda x: len(x[1]))
    # sort_s = [x[1] for x in sort_arr]
    # result = []
    # for i in tqdm(range(0, len(sort_s), 2048)):
    #     result.extend(HanLP(sort_s[i:i+2048])['ner'])
    # result = [(x[0], x[2]) for x in zip(sort_arr, result)]
    # result.sort(key=lambda x: x[0])
    # print([x[0] for x in result[:40]])
    # return [x[1] for x in result]


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

def main():
    csv.field_size_limit(500 * 1024 * 1024)
    env = lmdb.open('/mnt/data/mzc/key', map_size = 1099511627776)

    HanLP = hanlp.load(hanlp.pretrained.mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    tasks = list(HanLP.tasks.keys())
    for task in tasks:
        if task not in ('tok', 'ner'):
            del HanLP[task]
    for task in HanLP.tasks.values():
        task.sampler_builder.batch_size = 64

    cnt = 0
    errcnt = 0
    contentList = glob.glob('/mnt/data/mysql_data/*.csv')
    # print(contentList)
    for a in tqdm(contentList):

        basename = os.path.basename(a)
        txn = env.begin(write=True)

        print(basename, cnt, errcnt)

        plain_text_arr = []
        lines = []

        fcsv = csv.reader(open(a, 'r', encoding='utf-8'))
        flag = False
        for row in tqdm(fcsv):
            if not flag:
                flag = True
                continue
            try:
                qwContent = magic(row[1])
            except BaseException:
                errcnt += 1
                print('ERROR', a, cnt)
                with open('error', 'w', encoding='UTF-8') as f:
                    f.write(row[1])
                
            if qwContent is not None:
                cnt += 1
                plain_text = extract_plaintext(qwContent)
                l = len(plain_text_arr)
                plain_text_arr.extend(plain_text)
                r = len(plain_text_arr)
                # key_words = get_key(HanLP, plain_text)
                lines.append([str(cnt), row[0], row[2], basename, l, r])
                # txn.put(str(cnt).encode(), json.dumps( [row[0], row[2], basename, key_words] ).encode() )
                # break
        # print('ok')
        key_arr = get_key(HanLP, plain_text_arr)
        # print(key_arr)
        for line in lines:
            count, docId, Id, basename, l, r = line
            key_words = calc(key_arr[l:r])
            # print(l, r, key_words)
            txn.put(count.encode(), json.dumps( [docId, Id, basename, key_words] ).encode() )
        # exit()
        
        txn.commit()
        # break

    print(cnt, errcnt)
    env.close()

if __name__ == '__main__':

    main()

    # with open('error','r') as f:
    #     print(magic(f.read(-1)))

    # env = lmdb.open('/mnt/data/mzc/main.db', map_size=1099511627776)
    # txn = env.begin(write=True)
    # print(json.loads(txn.get('1'.encode()).decode()))
    # env.close()