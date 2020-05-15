#!/bin/sh

lang_pair=${@:1}
source=${lang_pair##*-}
target=${lang_pair%-*} 

echo source $source
echo target $target

vocab="../data/vocab_$lang_pair.bin"
train_src="../data/train.$lang_pair.$source.txt"
train_tgt="../data/train.$lang_pair.$target.txt"
dev_src="../data/dev.$lang_pair.$source.txt"
dev_tgt="../data/dev.$lang_pair.$target.txt"
test_src="../data/test.$lang_pair.$source.txt"
test_tgt="../data/test.$lang_pair.$target.txt"
test_tgt="../data/test.$lang_pair.en.txt"
embed_file="../wiki.$source.vec"

work_dir="results/$lang_pair"

mkdir -p ${work_dir}
echo save re

# comment below to test the decoder

python ../models/nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 800 \
    --batch-size 16 \
    --hidden-size 512 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --num-layers 1 \
    --attention-type 'general' \
    --bidirectional \
    --embedding_file ${embed_file}

# FOR BIDIRECTIONAL add the flag --bidirectional


#python nmt.py \
#    decode \
#    --cuda \
#    --beam-size 5 \
#    --max-decoding-time-step 100 \
#    ${work_dir}/model.bin \
#    ${test_src} \
#    ${work_dir}/decode.txt

#perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
