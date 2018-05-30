SHELL_FOLDER=$(dirname "${0}")
python3 -u ${SHELL_FOLDER}/../src/eval.py -d AISpeech -m focus -e 256 -l 1 -hs 256 -c GRU --post_attention  --bidirection --annotation 
#python3 -u ${SHELL_FOLDER}/../src/eval.py -d SpeechLab -e 256 --bidirection -c GRU -l 1 -hs 512
