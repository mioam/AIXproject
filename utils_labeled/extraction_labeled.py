import hanlp
import torch
import lmdb
import json
from tqdm import tqdm
from utils.extraction import NER, extract_plaintext, get_key, calc

def main():
    ner = NER()
    texts = torch.load('/mnt/data/mzc/datasets/pre/content.pt')
    # texts = texts[:10]
    plain_text_arr = []
    lines = []
    for i, qwContent in enumerate(texts):
        plain_text = extract_plaintext(qwContent)
        l = len(plain_text_arr)
        plain_text_arr.extend(plain_text)
        r = len(plain_text_arr)
        lines.append([str(i), i, i, 'content.pt', l, r])
    key_arr = get_key(ner, plain_text_arr)

    env = lmdb.open('/mnt/data/mzc/key_tmp', map_size = 1099511627776)
    txn = env.begin(write=True)

    for line in lines:
        count, docId, Id, basename, l, r = line
        key_words = calc(key_arr[l:r])
        # print(l, r, key_words)
        txn.put(count.encode(), json.dumps( [docId, Id, basename, key_words] ).encode() )
    # exit()

    txn.commit()
    env.close()

    # torch.save(texts,'datasets/feature/entity.pt')


if __name__ == "__main__":
    main()



