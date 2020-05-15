### Sentence Evaluation Experiments

#### Pre-run Requirements:
-------------------------------
* Python 3.6 environment
* wget https://github.com/facebookresearch/SentEval/archive/master.zip
* unzip master.zip
* mv SentEval-master SentEval
-------------------------------

### Steps to Replicate

#### Step 1

```wget http://nlp.stanford.edu/data/glove.6B.zip```

```unzip glove.6B.zip```

#### Step 2

```python dim.py 300 glove300d```

#### Step 3

```Choose a task in bow_main.py```

#### Step 4

```bash run.sh >& results_bigram.txt```

#### Step 5

```python extract_single_task.py results_bigram.txt```
