wget https://github.com/facebookresearch/fastText/archive/master.zip
unzip master.zip
cd fastText-master/
make
cd ..
mv train_fasttext.sh fastText-master/
rm -f master.zip
cd fastText-master/
