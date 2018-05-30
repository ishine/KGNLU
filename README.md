##KGNLU
Natural Language Understanding over Knowledge Graph.

##目录结构
> data/: training and dev dataset generated from fst rules
>> AISpeech/: domain dir
>>> rules.txt: designed rule templates for NLU
>>> constraints.txt: constraints for NLU
>>> constraints_mappings.txt: mappings from constraint to mapped value
>>> rules.release.txt: release pattern regex in rules.txt, one fst rule extended to many instances or paths
>>> rules.annotation.txt: replace entities and constraints in rules.release.txt according to the built fsts, plus generating annotation
>>> dict.txt: dict file for pyltp token segmentor
>>> ontology.json: ontology of current domain in json format
>>> fst_src/: dir to store built fsts of lexicons and constraints
>>> tagger_4fst/: lexicon dir, including entity names for each concept, used to build lexicon fst
>>> postprocess/: postprocess entities normalization
>>> rulels.test.*: files used to generate test questionnaire sentences
>> SpeechLab/: domain dir, the same structure as AISpeech
> scripts/: shell .sh scripts
>> buildFst.sh: build fsts for lexicons(taggers) and constraints
>> releaseRegex.sh: release pattern regex in rules.txt and generate rules.release.txt
>> addAnnotation.sh: replace entities and constraints in rules.release.txt and add annotations, generating rules.annotation.txt and sentence/slot tags
>> generateData.sh: combine buildFst.sh, releaseRegex.sh and addAnnotation.sh
>> transformDataFormat.sh: convert rules.annotation.txt to training data format
>> tran.sh: train the network and save best model
>> eval.sh: given test file, generating annotation result
>> collect_testdata.py: generate test set original sentences
> res: resources after preprocessing training data,including voc2idx, slottag2idx, sentencetag2idx, train.txt, test.txt, eval.txt and stop_pattern.txt
> src: source code
>> train.py: main function to train and test the model
>> utils.py: some functions used during preprocessing and postprocessing
>> global_vars: defines global file paths
>> gpu_selection.py: used to select gpu when using cuda
>> eval.py: main function to eval sentences and generate parsed result
>> collect_test_data/: dir containing functions to generate testset
>> prepare_training_data/: dir containing functions to generate training set
>> models: dir containing models, Bi-RNN, Bi-RNN+CRF, Focus and Attention Models, decoder functions, evaluation functions and loss functions
> model: dir to save models

##
