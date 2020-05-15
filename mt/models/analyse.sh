wget https://github.com/neulab/compare-mt/blob/master/compare-mt.py
python compare-mt.py data/test.de-en.en decode.txt > mt_single_results.txt
python compare-mt.py data/test.de-en.en decode_512_1layer_bi.txt decode_512_3layer_bi_general.txt > compare_new.txt

wget http://www.cs.cmu.edu/~jhclark/downloads/multeval-0.5.1.tgz
tar -xvzf multeval-0.5.1.tgz
cd multeval-0.5.1
./multeval.sh eval --refs ../data/test.de-en.en  --hyps-baseline ../decode.txt  --meteor.language en > multeval_compare_results.txt
./multeval.sh eval --refs ../data/test.de-en.en  --hyps-baseline ../decode_512_1layer_bi.txt --hyps-sys1 ../decode_512_3layer_bi_general.txt --meteor.language en > compare_multeval.txt
