### Machine Translation Experiments

#### Requirements:
-------------------------------
* Python 3.6 environment
* ```pip install -r requirements.txt```
-------------------------------

### Commands to get embeddings

```get_all_pretrained.sh```

### Commands to Train + Evaluate (e.g. a language pair might be: en-az):

```cd scripts```

```bash build_vocab.sh en-az```

```bash train.sh en-az```

```bash decode.sh en-az```
