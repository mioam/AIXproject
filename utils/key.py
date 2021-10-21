import lmdb
import json
from utils import build_map

if __name__ =='__main__':
    env = lmdb.open('/mnt/data/mzc/key')
    env_map = lmdb.open('/mnt/data/mzc/map')

    txn = env.begin(write=False)
    txn_map = env_map.begin(write=False)
    # print(txn.stat())
    # print(txn_map.stat())

    # print(json.loads(txn.get('1'.encode())))
    # print(json.loads(txn.get('11587'.encode())))
    # print(json.loads(txn.get('11639'.encode())))
    print(json.loads(txn.get('19198'.encode())))
    print(json.loads(txn.get('19033'.encode())))

    env.close()
    env_map.close()