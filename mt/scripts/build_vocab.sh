lang_pair=${@:1}
source=${lang_pair##*-}
target=${lang_pair%-*}

echo $source
echo $target

python3 ../models/vocab.py --train-src ../data/train.$lang_pair.$source.txt  --train-tgt ../data/train.$lang_pair.$target.txt --vocab-type 'word' --size 50000 ../data/vocab_$lang_pair.bin 
