for i in {1..299}; do 
     echo python bow.py "$i"
     python bow.py  1 ./glove_300_embedding_"$i".txt
 done
