#%%
import pickle
import json
import re
import ast
import pandas as pd

from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt
from sklearn_crfsuite import CRF

from model.crf_utils import text2features, Custom_CRF
#%%
DEV_QUESTIONS_FILENAME = "data/preprocessedData/dev_questions.csv"
MODEL_PREDICTIONS_FILENAME = "data/model_predictions/BERT (single model) (Google AI Language).json"
MODEL_FILENAME = "model/trainedModels/crf_sample_1000.obj"

POS_FEATURES = True
ENT_TYPE_FEATURES = True
LEMMA_FEATURES = True
IS_FEATURES = True
POSITION_FEATURES = True
BIAS = True
BEGIN = -1
END = 1

#%%
DF_MODEL_PREDICTIONS = pd.DataFrame()
with open(MODEL_PREDICTIONS_FILENAME) as json_data:
    JSON_DICT = json.load(json_data)
    for question_id in JSON_DICT.keys():
        DF_MODEL_PREDICTIONS = DF_MODEL_PREDICTIONS.append({"question_id": question_id,
                                                            "answer": JSON_DICT[question_id]},
                                                           ignore_index=True)
DF_MODEL_PREDICTIONS.head()

#%%
DF_DEV_QUESTIONS = pd.read_csv(DEV_QUESTIONS_FILENAME, index_col="question_id")
DF_DEV_QUESTIONS.head()

#%% Load saved model for prediction of question-worthy tokens
with open(MODEL_FILENAME, 'rb') as f:
    crf = pickle.load(f)

#%%
def char_index_2_token_index(paragraph, substring_start, substring_text):
    doc = nlp(paragraph)
    span = doc.char_span(substring_start, substring_start+len(substring_text))

    try:
        return span.start, span.end
    except AttributeError:
        return (None, None)

def get_likelihoods(paragraph_text, correct_answer_token_indices, answer_token_indices):
    paragraph = nlp(paragraph_text)
    paragraph_features = text2features(paragraph,
                                       pos_features=POS_FEATURES,
                                       ent_type_features=ENT_TYPE_FEATURES,
                                       lemma_features=LEMMA_FEATURES,
                                       is_features=IS_FEATURES,
                                       position_features=POSITION_FEATURES,
                                       bias=BIAS,
                                       begin=BEGIN,
                                       end=END)
    predictions = [token["I"] for token in crf.predict_marginals_single(paragraph_features)]

    correct_answer_likelihoods = []
    for correct_answer_token_index in set(correct_answer_token_indices):
        correct_answer_token_start, correct_answer_token_end = correct_answer_token_index
        correct_answer_likelihoods.append(max(predictions[correct_answer_token_start:correct_answer_token_end]))

    answer_likelihoods = []
    for answer_token_index in set(answer_token_indices):
        answer_token_start, answer_token_end = answer_token_index
        answer_likelihoods.append(max(predictions[answer_token_start:answer_token_end]))

    return max(correct_answer_likelihoods, default=None), max(answer_likelihoods, default=None)
    

def string_to_token_index(paragraph, substring):
    if substring == "":
        return []
    else:
        result = []
        for match in re.finditer(re.escape(substring), paragraph):
            index = char_index_2_token_index(paragraph, match.start(), substring)
            if not index == (None, None):
                result.append(index)
        return result

def find_property_to_questionid(question_id, property_name):
    try:
        return DF_DEV_QUESTIONS.loc[question_id][property_name]
    except KeyError:
        return None

def substring_match(correct_answer_texts, answer_text):
    if answer_text == "":
        return False
    return any((answer_text in correct_answer_text) or (correct_answer_text in answer_text) for correct_answer_text in correct_answer_texts)

def complete_match(correct_answer_texts, answer_text):
    if answer_text == "":
        return False
    return answer_text in correct_answer_texts

#%%
nlp = spacy.load('en')

text = "A Japan-exclusive manga series based on Twilight Princess, penned and illustrated by Akira Himekawa, was first released on February 8, 2016. The series is available solely via publisher Shogakukan's MangaOne mobile application. While the manga adaptation began almost ten years after the initial release of the game on which it is based, it launched only a month before the release of the high-definition remake."
y_true = ['O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

assert get_likelihoods(text, [(16, 18)], [(16, 18)])[0] == 0.4522455631369173
assert string_to_token_index(text, "Twilight Princess") == [(8, 10)]
assert find_property_to_questionid("56ddde6b9a695914005b9628", "paragraph_context")[:11] == "The Normans"

#%% add the indicies, the correct answer text and the paragraph to the df
DF_MODEL_PREDICTIONS["paragraph"] = [find_property_to_questionid(question_id, "paragraph_context") for question_id in DF_MODEL_PREDICTIONS["question_id"]]
DF_MODEL_PREDICTIONS["correct_answer_text"] = [ast.literal_eval(find_property_to_questionid(question_id, "correct_answer_text")) for question_id in DF_MODEL_PREDICTIONS["question_id"]]
DF_MODEL_PREDICTIONS["correct_answer_token_index"] = [ast.literal_eval(find_property_to_questionid(question_id, "correct_answer_token_index")) for question_id in DF_MODEL_PREDICTIONS["question_id"]]
DF_MODEL_PREDICTIONS["answer_token_index"] = [string_to_token_index(paragraph, answer) if not paragraph is None else [] for paragraph, answer in tqdm(zip(DF_MODEL_PREDICTIONS["paragraph"], DF_MODEL_PREDICTIONS["answer"]))]
DF_MODEL_PREDICTIONS.head()

#%% calculate the likelihood of the answer and the correct answer
all_answer_likelihoods = []
all_correct_answer_likelihood = []
for paragraph, correct_answer_token_indices, answer_token_indices in tqdm(zip(DF_MODEL_PREDICTIONS["paragraph"], DF_MODEL_PREDICTIONS["correct_answer_token_index"], DF_MODEL_PREDICTIONS["answer_token_index"])):
    correct_answer_likelihood, answer_likelihood = get_likelihoods(paragraph, correct_answer_token_indices, answer_token_indices)
    all_correct_answer_likelihood.append(correct_answer_likelihood)
    all_answer_likelihoods.append(answer_likelihood)

DF_MODEL_PREDICTIONS["answer_likelihood"] = all_answer_likelihoods
DF_MODEL_PREDICTIONS["correct_answer_likelihood"] = all_correct_answer_likelihood


#%%
DF_MODEL_PREDICTIONS.head()


#%%
DF_MODEL_PREDICTIONS["complete_match"] = [complete_match(correct_answer_text, answer) for correct_answer_text, answer in zip(DF_MODEL_PREDICTIONS["correct_answer_text"], DF_MODEL_PREDICTIONS["answer"])]

#%%
DF_MODEL_PREDICTIONS["substring_match"] = [substring_match(correct_answer_text, answer) for correct_answer_text, answer in zip(DF_MODEL_PREDICTIONS["correct_answer_text"], DF_MODEL_PREDICTIONS["answer"])]


#%%
DF_MODEL_PREDICTIONS.head(50)




#%%
DF_MODEL_PREDICTIONS_DROPNA = DF_MODEL_PREDICTIONS.dropna(subset=["answer_likelihood", "correct_answer_likelihood"])

DF_MODEL_PREDICTIONS_correct = DF_MODEL_PREDICTIONS_DROPNA[DF_MODEL_PREDICTIONS_DROPNA["substring_match"] == True]
DF_MODEL_PREDICTIONS_wrong = DF_MODEL_PREDICTIONS_DROPNA[DF_MODEL_PREDICTIONS_DROPNA["substring_match"] == False]

#%%
DF_MODEL_PREDICTIONS_correct
#%%
DF_MODEL_PREDICTIONS_wrong
#%%
DF_MODEL_PREDICTIONS_DROPNA

#%%
plt.boxplot([DF_MODEL_PREDICTIONS_correct["answer_likelihood"], DF_MODEL_PREDICTIONS_wrong["answer_likelihood"], DF_MODEL_PREDICTIONS_correct["correct_answer_likelihood"], DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"]])


#%%
plt.scatter(DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"], DF_MODEL_PREDICTIONS_wrong["answer_likelihood"], c=DF_MODEL_PREDICTIONS_wrong["substring_match"])
plt.plot([0, 0.8], [0, 0.8], color='k', linestyle='-', linewidth=2)


#%%
plt.scatter(DF_MODEL_PREDICTIONS_DROPNA["correct_answer_likelihood"], DF_MODEL_PREDICTIONS_DROPNA["answer_likelihood"], c=DF_MODEL_PREDICTIONS_DROPNA["complete_match"])


#%%
plt.scatter(DF_MODEL_PREDICTIONS_DROPNA["correct_answer_likelihood"], DF_MODEL_PREDICTIONS_DROPNA["answer_likelihood"], c=DF_MODEL_PREDICTIONS_DROPNA["substring_match"])

#%%
plt.hist(DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"], 25,facecolor='blue', alpha=0.5)
plt.show()
plt.hist(DF_MODEL_PREDICTIONS_correct["correct_answer_likelihood"], 25,facecolor='blue', alpha=0.5)
plt.show()

#%%
plt.hist(DF_MODEL_PREDICTIONS_wrong["answer_likelihood"], 25,facecolor='blue', alpha=0.5)
plt.show()
plt.hist(DF_MODEL_PREDICTIONS_correct["answer_likelihood"], 25,facecolor='blue', alpha=0.5)
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(DF_MODEL_PREDICTIONS_wrong["answer_likelihood"], 25, facecolor='blue', alpha=0.5)
ax.hist(DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"], 25, facecolor='blue', alpha=0.5)
plt.xlabel('(correct) answer likelihood')
plt.title('Histogram of (correct) Answer Likelihood for wrongly predicted Answers')
plt.savefig('hist_wrong_answer_likelihood.png')
plt.show()

#%%
DF_MODEL_PREDICTIONS_correct.head()
DF_MODEL_PREDICTIONS_wrong.head()