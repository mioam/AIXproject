import lmdb
import json
from tqdm import tqdm

def get_words(value):
    key_words = value[3]
    return [word 
    for Type, Words in key_words.items()
        for word, count in Words]


Threshold = 100

def map_key(word, number):
    return (str(number).zfill(4) + word).encode()

def add(key, word, env_map, txn_map):
    # txn_map = env_map.begin(write=True)
    # print(word)
    if txn_map.get(map_key(word, Threshold-1)) is None:
        for i in range(Threshold):
            if txn_map.get(map_key(word, i)) is None:
                txn_map.put(map_key(word, i), key)
                break

    # txn_map.commit()

def main():
    env = lmdb.open('/mnt/data/mzc/key')
    env_map = lmdb.open('/mnt/data/mzc/map', map_size=1099511627776)
    txn = env.begin(write=False)
    txn_map = env_map.begin(write=True)

    print(txn.stat())
    for key, value in tqdm(txn.cursor()):
        value = json.loads(value)
        # print (key, value)
        key_words = value[3]
        for word in get_words(value):
            add(key, word, env_map, txn_map)
        # break
    txn_map.commit()

    # print(json.loads(txn.get('10'.encode())))
    env_map.close()
    env.close()

if __name__=='__main__':
    main()

    # 2000it/s