import lmdb
import json
from utils import build_map

if __name__ =='__main__':
    # env = lmdb.open('/mnt/data/mzc/key')
    # env_map = lmdb.open('/mnt/data/mzc/map')

    # txn = env.begin(write=False)
    # txn_map = env_map.begin(write=False)
    # # print(txn.stat())
    # # print(txn_map.stat())

    # # print(json.loads(txn.get('1'.encode())))
    # # print(json.loads(txn.get('11587'.encode())))
    # # print(json.loads(txn.get('11639'.encode())))
    # print(json.loads(txn.get('321323'.encode())))
    # print(json.loads(txn.get('321318'.encode())))
    # print(json.loads(txn.get('321321'.encode())))
    # print(json.loads(txn.get('347225'.encode())))
    # print(json.loads(txn.get('321314'.encode())))
    # print(json.loads(txn.get('321322'.encode())))

    # env.close()
    # env_map.close()
    # exit()

    env = lmdb.open('/mnt/data/mzc/key_tmp')
    env_map = lmdb.open('/mnt/data/mzc/map_tmp')

    txn = env.begin(write=False)
    txn_map = env_map.begin(write=False)
    print(txn.stat())
    print(txn_map.stat())

    print(json.loads(txn.get('12345'.encode())))
    print(json.loads(txn.get('216357'.encode())))
    print(json.loads(txn.get('119418'.encode())))
    # print(txn_map.get('0000黄浦区'.encode()))

    env.close()
