echo "EN-AZ Started"

bash build_vocab.sh en-azen > en-azen.vocab.txt
bash train.sh en-azen > en-azen.train.txt
bash decode.sh en-azen > en-azen.decode.txt

echo "EN-BE Started"

bash build_vocab.sh en-been > en-been.vocab.txt
bash train.sh en-been > en-been.train.txt
bash decode.sh en-been > en-been.decode.txt

echo "EN-GL Started"

bash build_vocab.sh en-glen > en-glen.vocab.txt
bash train.sh en-glen > en-glen.train.txt
bash decode.sh en-glen > en-glen.decode.txt
