for i in {1..299}; do
     echo python bow_main.py "$i"
     python bow_main.py  1 ../../single_glove_300_embedding_"$i".pickle
done
