
# To use, clone WikiExtractor (https://github.com/attardi/wikiextractor) and Moses (https://github.com/moses-smt/mosesdecoder) and place in this directory. Then run with:
# sh wikipedia.sh <language> where <language> is a two-letter language code.
# Example: sh wikipedia.sh gl

LANGUAGE=$1

#download wikipedia dump
#wget https://dumps.wikimedia.org/${LANGUAGE}wiki/20181001/${LANGUAGE}wiki-20181001-pages-articles-multistream.xml.bz2 .
wget https://dumps.wikimedia.org/enwikinews/20181101/enwikinews-20181101-pages-articles-multistream.xml.bz2 .

#extract dump
bzip2 -d ${LANGUAGE}wikinews-20181101-pages-articles-multistream.xml.bz2

#extract articles in one big file
cd $WIKI_EX
python WikiExtractor.py -b 100000000000 ../${LANGUAGE}wikinews-20181101-pages-articles-multistream.xml -o ..
cd ..
mv AA/wiki_00 ${LANGUAGE}.wiki.txt
rm -Rf AA

#remove blank lines and doc tags
awk 'NF' ${LANGUAGE}.wiki.txt | grep -v "^<" > ${LANGUAGE}.wiki.temp.txt
mv ${LANGUAGE}.wiki.temp.txt ${LANGUAGE}.wiki.txt

#tokenize
cat ${LANGUAGE}.wiki.txt | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${LANGUAGE} > ${LANGUAGE}.wiki.tok.txt
mv ${LANGUAGE}.wiki.tok.txt ${LANGUAGE}.wiki.txt
