#!/bin/sh

vocab="data/vocab.bin"
train_src="../data/train.en-az.en.txt"
train_tgt="../data/train.en-az.az.txt"
dev_src="../data/dev.en-az.en.txt"
dev_tgt="../data/dev.en-az.az.txt"
test_src="../data/test.en-az.en.txt"
test_tgt="../data/test.en-az.az.txt"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save re

# comment below to test the decoder

python nmt.py \
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
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --num-layers 2 \
    --attention-type 'general' \
    --bidirectional

# FOR BIDIRECTIONAL add the flag --bidirectional


python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
