# AIXproject

### 预处理

- `./preprocess.py` (已废弃)
  - 数据预处理。把 `./datasets/json/content[0-9]*.json` 和 `./datasets/json/0.json` 中有用的对提出来，relabel，tokenize。炸内存。
- `./pre1_relabel.py`
  - 数据预处理。把 `./datasets/json/content[0-9]*.json` 和 `./datasets/json/0.json` 中有用的对提出来，relabel。
  - `./datasets/pre/content.pt`
    - list，每个元素是文本内容，字符串。
  - `./datasets/pre/relation.pt`
    - list，每个元素是tuple (x,y)，表示第x和第y篇有联系。
    - 是原始的相关文书标签relabel
    - 删除了和自身的联系
  - `./datasets/pre/labeledId.pt`
    - dict，labeledId[docID] = index
- `./pre2_tokenizer.py`
  - 把 `./datasets/pre/content.pt` 中的html标签删除，tokenize。使用 BertTokenizerFast 和 BeautifulSoup。
  - `./datasets/pre/content_tokens/{i}.pt`
    - list，处理完了的token。


#### NER_IDCNN_CRF

requirement:

cuda10? https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/

tensorflow==1.15 jieba beautifulsoup4 jionlp

virtualenv: tf1

- `./NER_IDCNN_CRF\my_extraction.py`
  - 读取 `./datasets/pre/content.pt`，用 BeautifulSoup 提取文本，用jionlp分句，用 https://github.com/crownpku/Information-Extraction-Chinese 进行预处理。
  - 忽略以 ('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员') 开头的行及之后的行。
  - `./datasets/feature/entity.pt` 保存对应文本的LOC，ORG，PER。词和次数的tuple。
  - NER不够行，会有逗号。会有不低的错误率。
  - 有些词应当删除，如“中华人民共和国”。
- `./NER_IDCNN_CRF\only_soup.py`
  - 只有 BeautifulSoup。试验代码。



### 加载数据集

- `./dataset.py`
  - FeatureDataset
  - WenshuDataset
