#coding=utf8
import sys,os

PWD=os.path.dirname(__file__)

PATH_TO_ONTOLOGY={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/ontology.json'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/ontology.json')
}

PATH_TO_MODELS={
    'AISpeech':os.path.join(PWD,'../model/AISpeech/'),
    'SpeechLab':os.path.join(PWD,'../model/SpeechLab/')
}

PATH_TO_VOC={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/vocabulary.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/vocabulary.txt'),    
}

PATH_TO_SLOT_TAGSET={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/slot_tagset.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/slot_tagset.txt'),
}

PATH_TO_SENTENCE_TAGSET={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/sentence_tagset.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/sentence_tagset.txt')
}

PATH_TO_TRAINING_DATA={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/train.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/train.txt')
}

PATH_TO_TEST_DATA={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/test.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/test.txt')
}

PATH_TO_EVAL_DATA={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/eval.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/eval.txt')    
}

PATH_TO_DATA={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/')
}

PATH_TO_SPLIT_WORDS=os.path.join(PWD,'../data/cws.model')

PATH_TO_STOP_PATTERN={
    'AISpeech':os.path.join(PWD,'../res/AISpeech/stop_pattern.txt'),
    'SpeechLab':os.path.join(PWD,'../res/SpeechLab/stop_pattern.txt')      
}

PATH_TO_FST={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/fst_src/'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/fst_src/')
}

PATH_TO_TAGGER={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/tagger_4fst/'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/tagger_4fst/')
}

PATH_TO_CONSTRAINT={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/constraints.txt'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/constraints.txt')
}

PATH_TO_CONSTRAINT_MAPPINGS={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/constraint_mappings.txt'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/constraint_mappings.txt')    
}

PATH_TO_POST_PROCESS_DIR={
    'AISpeech':os.path.join(PWD,'../data/AISpeech/postprocess/'),
    'SpeechLab':os.path.join(PWD,'../data/SpeechLab/postprocess/')     
}