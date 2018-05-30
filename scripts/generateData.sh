SHELL_FOLDER=$(dirname "${0}")
if [ -n "${1}" ];then
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/fst_builder.py -d ${1} >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/release_regex.py -d ${1} >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/annotation_adder.py -d ${1} >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/transform_dataset.py -d ${1}
else
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/fst_builder.py -d AISpeech >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/fst_builder.py -d SpeechLab >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/release_regex.py -d AISpeech >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/release_regex.py -d SpeechLab >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/annotation_adder.py -d AISpeech >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/annotation_adder.py -d SpeechLab >> /dev/null
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/transform_dataset.py -d AISpeech
    python3 ${SHELL_FOLDER}/../src/prepare_training_data/transform_dataset.py -d SpeechLab
fi