f = open("wikis/az.wiki.txt")
documents = []

for line in f:
    documents.append(line.split())

print(len(documents)) 

import gensim
import subprocess

# build vocabulary and train model
model = gensim.models.Word2Vec(
        documents,
        sg=1, # 0 means cbow
        size=256,
        window=10,
        min_count=1, # make min count = 1
        workers=10)

model.train(documents, total_examples=len(documents), epochs=10)

embedding_file = open('w2v_embedding.txt', 'w')

words = list(model.wv.vocab)
print(len(words)) 

for i, x in enumerate(words):
    #print(i, x, model[x])
    embedding_file.write("%s\t" % x)
    for t in model[x]:
            embedding_file.write("%f\t" % float(t))        
    embedding_file.write("\n")

embedding_file.close()
