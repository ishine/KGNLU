SHELL_FOLDER=$(dirname "${0}")
if [ -n "${1}" ];then
    python3 -u ${SHELL_FOLDER}/../src/train.py -m rnn --debug --train --validation -d $1 -e 256 -c LSTM -hs 512 --bidirection -l 1 -es 1  > ${SHELL_FOLDER}/../train_${1}_attention_LSTM_256_512.log 
else
    python3 -u ${SHELL_FOLDER}/../src/train.py --debug --validation -d AISpeech -e 256 -c LSTM -hs 512 --bidirection -l 1 -es 5 --cuda > ${SHELL_FOLDER}/../train_AISpeech.log 
    python3 -u ${SHELL_FOLDER}/../src/train.py --debug --validation -d SpeechLab -e 256 -c LSTM -hs 512 --bidirection -l 1 -es 5 --cuda > ${SHELL_FOLDER}/../train_SpeechLab.log
fi
