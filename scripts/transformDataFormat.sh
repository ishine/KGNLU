SHELL_FOLDER=$(dirname "${0}")
if [ -n "${1}" ];then
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/transform_dataset.py -d ${1} 
else
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/transform_dataset.py -d AISpeech
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/transform_dataset.py -d SpeechLab
fi
