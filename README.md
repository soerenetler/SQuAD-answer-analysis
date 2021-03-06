# SQuAD-answer-analysis

## Requirements

- matplotlib
- sklearn_crfsuite
- graphviz
- scikit-learn == 0.19.1 (this is a slightly older version that works with sklearn_crfsuite)
- pandas
- tqdm
- spacy
- pickle

## Preprocessing
The data is preprocessed in the preprocessing.py script. It keeps the original train / dev split. The raw Date is expected in the folder data/rawData with the dev-v2.0.json and the train-v2.0.json.

In the process of preprocessing invalid_spans are saved in the invalid_spans.csv file. This are cases where the char index of the answer span could not be mapped to a valid token in the text. This might be the case, because of errors in the tokenization.

A very brief analysis of the dataset is done in the dataset_analysis notebook.

## Identifying question-worthy tokens

All files regarding the identificytion of question-worthy tokens can be found in the model folder.

The crf_utils.py and the evaluation_utils.py provide helperfunctions that make the training and evaluation process easier and the code in the notebooks cleaner.

The trainCRF.py runs the training process (this can take a while to run) and saves the trained model with the tuned hyperparameters in teh trainedModels folder.

The evaluation folder contains the files for the different baselines the model is compared to.

The evaluation of the crf models can be found in the root folder of the repository: evaluate_crf.ipynb. This file runs very fast and illustrates the models.

## Evaluation of QA models

The evaluate_qa_utils.ipynb runs the evaluation of one QA model. The downoaded predictions of those models are saved in the data/model_predictions folder. The helper functions are listet in the evaluate_qa_utils.py
