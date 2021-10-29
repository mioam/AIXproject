# AIXproject

### 预处理

- `./utils/t/preprocess.py` (已废弃)
  - 数据预处理。把 `./datasets/json/content[0-9]*.json` 和 `./datasets/json/0.json` 中有用的对提出来，relabel，tokenize。炸内存。
- `./utils/t/pre1_relabel.py`
  - 数据预处理。把 `./datasets/json/content[0-9]*.json` 和 `./datasets/json/0.json` 中有用的对提出来，relabel。
  - **`./datasets/pre/content.pt`**
    - list，每个元素是文本内容，字符串。
  - **`./datasets/pre/relation.pt`**
    - list，每个元素是tuple (x,y)，表示第x和第y篇有联系。
    - 是原始的相关文书标签relabel
    - 删除了和自身的联系
  - `./datasets/pre/labeledId.pt`
    - dict，labeledId[docID] = index
- `./utils/t/pre2_tokenizer.py`
  - 把 `./datasets/pre/content.pt` 中的html标签删除，tokenize。使用 BertTokenizerFast 和 BeautifulSoup。
  - `./datasets/pre/content_tokens/{i}.pt`
    - list，处理完了的token。

#### Utils

- `./utils/extraction.py`
  - 从csv提NER关键字，存到 `/mnt/data/mzc/key`。
  - magic: 从csv的值变成html
  - split_sentence：断句
  - extract_plaintext：从html变成纯文本，然后调split_sentence断句
  - get_key：跑NER
  - calc：把NER的结果整理合并
  - work：对csv文件，搞
  - getNO：提取csv文件名中的数字
- `./utils/build_map.py`
  - 把 `/mnt/data/mzc/key`中的关键字建个索引，存到 `/mnt/data/mzc/map`。
  - get_words：从calc返回值到关键词list
  - map_key：把词和编号，变成字符串编码。第13个'张三' -> '0013张三'
  - add：尝试加一个关键词。同一个关键词最多有Threshold 100个
- `./utils/lmdb_get.py`
  - 对于给定的文书，查找相关文书。
  - candidate
- `./utils/key.py`
  - 对于给定的编号，查找文书关键字。

#### For Labeled Set

- `./utils_labeled/getBERT.py`
  - 读取 `./datasets/pre/content.pt`，用 BeautifulSoup 提取文本，获得bert输出。Sliding window。mean。
  - **`./datasets/feature/bert.pt`** 保存对应文本的bert输出。
    - 空文本（187632）。现在会返回0向量。
- `./utils_labeled/getNER.py` (待修改)
  - 读取 `./datasets/pre/content.pt`，用 hanlp 获得关键词。
- `./utils_labeled/extraction_labeled.py`
  - `./utils/extraction.py` 在labeled set上的版本。
- `./utils_labeled/lmdb_get_labeled.py`
  -  关键的处理代码
- `./utils_labeled/content.py`
  - 对于给定的编号，查找文书。




### 加载数据集

- `./dataset.py`
  - FeatureDataset
- `./train.py`

## 全数据集测试

TBD

1. 从csv提取json -> 提取html -> 提取文本 -> 断句，NER -> 储存断句结果 
2. 找稀有的key，储存key的哈希表。
3. 输入文本 -> NER -> 检索key -> 提取文本 -> bert


## Demo

TBD

#### ~~NER_IDCNN_CRF~~

requirement:

cuda10? https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/

tensorflow-gpu==1.15 jieba beautifulsoup4 jionlp

virtualenv: tf1

command: `python my_extraction.py --ckpt_path=ckpt_IDCNN`

- `./NER_IDCNN_CRF/my_extraction.py`
  - 读取 `./datasets/pre/content.pt`，用 BeautifulSoup 提取文本，用jionlp分句，用 https://github.com/crownpku/Information-Extraction-Chinese 进行预处理。
  - 忽略以 ('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员') 开头的行及之后的行。
  - **`./datasets/feature/entity.pt`** 保存对应文本的LOC，ORG，PER。词和次数的tuple。
  - NER不够行，会有逗号。会有不低的错误率。
  - 有些词应当删除，如“中华人民共和国”。
- `./NER_IDCNN_CRF/only_soup.py`
  - 只有 BeautifulSoup。试验代码。
