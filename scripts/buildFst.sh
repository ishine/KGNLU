SHELL_FOLDER=$(dirname "${0}")
if [ -n "${1}" ];then
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/fst_builder.py -d ${1} 
else
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/fst_builder.py -d AISpeech
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/fst_builder.py -d SpeechLab
fi