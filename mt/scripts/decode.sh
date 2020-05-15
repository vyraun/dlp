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

work_dir="results/$lang_pair"

python ../models/nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
