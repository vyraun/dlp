### Machine Translation Experiments

#### Pre-Run

```pip install -r requirements.txt```

### Commands to get embeddings

```get_all_pretrained.sh```

### Commands to Train + Evaluate (e.g. a language pair might be: en-aztr):

```cd scripts```

```bash build_vocab.sh en-az```

```bash train.sh en-az```

```bash decode.sh en-az```
