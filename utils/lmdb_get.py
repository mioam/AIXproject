import lmdb
import json
from utils import build_map

if __name__ =='__main__':
    env = lmdb.open('/mnt/data/mzc/key')
    env_map = lmdb.open('/mnt/data/mzc/map')

    txn = env.begin(write=False)
    txn_map = env_map.begin(write=False)
    print(txn.stat())
    print(txn_map.stat())

    # for key, value in txn.cursor():
    key = '19198'
    value = txn.get(key.encode()).decode()
    for _ in [0]:
        value = json.loads(value)
        # print (key, value)
        key_words = value[3]
        candidate = {}
        for word in build_map.get_words(value):
            if txn_map.get(build_map.map_key(word,build_map.Threshold-1)) is not None:
                continue
            for i in range(build_map.Threshold):
                ret = txn_map.get(build_map.map_key(word,i))
                if ret is None:
                    break
                else:
                    # print(ret)
                    if ret in candidate:
                        candidate[ret] += 1
                    else:
                        candidate[ret] = 1
        candidate = list(candidate.items())
        candidate.sort(key=lambda x:-x[1])
        print(candidate)
        break

    env.close()
    env_map.close()