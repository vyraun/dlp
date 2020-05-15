#### Steps to replicate

The word-vector evaluation code is directly used from https://github.com/mfaruqui/eval-word-vectors.  

Run the script ```algo.py``` (embedding file location is hardcoded as of now) to reproduce the results on the dimensionality reduction algorithm and its evaluation. 

To run the algo and the baselines (as in the paper) get the embedding files ([Glove](https://nlp.stanford.edu/projects/glove/), [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)) or word2vec preprocessed in Glove format and put the file locations as required in the code.

The code will generate and evaluate (on 12 word-similarity datasets) a modified word embedding file that is half-the-size of the original embeddings.
