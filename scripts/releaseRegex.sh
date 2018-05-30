SHELL_FOLDER=$(dirname "${0}")
if [ -n "${1}" ];then
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/release_regex.py -d ${1}  >> /dev/null
else
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/release_regex.py -d AISpeech  >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/release_regex.py -d SpeechLab  >> /dev/null
fi
