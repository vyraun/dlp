#### Steps to replicate

The word-vector evaluation code is directly used from https://github.com/mfaruqui/eval-word-vectors.

First, get the embedding files ([Glove](https://nlp.stanford.edu/projects/glove/), [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)) and word2vec preprocessed in Glove format and put the file locations as required in the code ```pca_simple.py```.

Run the script ```pca_simple.py``` (embedding file location is hardcoded as of now) to reproduce the results on the PCA dimensionality reduction algorithm and its evaluation on word similarity tasks. 

Under default settings, the code will generate and evaluate (on 12 word-similarity datasets) a modified word embedding file that is half-the-size of the original embeddings. Please run it with different dimensions to get the corresponding embeddings.

Once the embedding is generated, please refer to sentence evaluation directory for running sentence classification experiments with these embeddings.
