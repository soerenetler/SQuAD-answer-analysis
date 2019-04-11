#%% Imports
import pickle
import json
import ast
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn_crfsuite import CRF

from model.crf_utils import Custom_CRF
from evaluate_qa_utils import get_likelihoods, string_to_token_index, find_property_to_questionid, substring_match, complete_match
#%% Parameter
DEV_QUESTIONS_FILENAME = "data/preprocessedData/dev_questions.csv"
MODEL_PREDICTIONS_FILENAME = "data/model_predictions/BiDAF + Self Attention + ELMo (single model) (Allen Institute for Artificial Intelligence [modified by Stanford]).json"
MODEL_FILENAME = "model/trainedModels/crf_sample_1000.obj"

#%% Load Model Predictions DF
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

#%% Load saved CRF model for prediction of question-worthy tokens
with open(MODEL_FILENAME, 'rb') as f:
    crf = pickle.load(f)
#%% Function Tests
text = "A Japan-exclusive manga series based on Twilight Princess, penned and illustrated by Akira Himekawa, was first released on February 8, 2016. The series is available solely via publisher Shogakukan's MangaOne mobile application. While the manga adaptation began almost ten years after the initial release of the game on which it is based, it launched only a month before the release of the high-definition remake."
y_true = ['O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

assert get_likelihoods(text, [(16, 18)], [(16, 18)], crf)[0] == 0.4522455631369173
assert string_to_token_index(text, "Twilight Princess") == [(8, 10)]
assert find_property_to_questionid("56ddde6b9a695914005b9628", "paragraph_context", DF_DEV_QUESTIONS)[:11] == "The Normans"

#%% add the indicies, the correct answer text and the paragraph to the df
DF_MODEL_PREDICTIONS["paragraph"] = [find_property_to_questionid(question_id, "paragraph_context", DF_DEV_QUESTIONS) for question_id in DF_MODEL_PREDICTIONS["question_id"]]
DF_MODEL_PREDICTIONS["correct_answer_text"] = [ast.literal_eval(find_property_to_questionid(question_id, "correct_answer_text", DF_DEV_QUESTIONS)) for question_id in DF_MODEL_PREDICTIONS["question_id"]]
DF_MODEL_PREDICTIONS["correct_answer_token_index"] = [ast.literal_eval(find_property_to_questionid(question_id, "correct_answer_token_index", DF_DEV_QUESTIONS)) for question_id in DF_MODEL_PREDICTIONS["question_id"]]
DF_MODEL_PREDICTIONS["answer_token_index"] = [string_to_token_index(paragraph, answer) if not paragraph is None else [] for paragraph, answer in tqdm(zip(DF_MODEL_PREDICTIONS["paragraph"], DF_MODEL_PREDICTIONS["answer"]))]
DF_MODEL_PREDICTIONS.head()

#%% calculate the likelihood of the answer and the correct answer
all_answer_likelihoods = []
all_correct_answer_likelihood = []
for index, row in tqdm(DF_MODEL_PREDICTIONS.iterrows()):
    paragraph = row["paragraph"]
    correct_answer_token_indices = row["correct_answer_token_index"]
    answer_token_indices = row["answer_token_index"]
    correct_answer_likelihood, answer_likelihood = get_likelihoods(paragraph, correct_answer_token_indices, answer_token_indices, crf)
    all_correct_answer_likelihood.append(correct_answer_likelihood)
    all_answer_likelihoods.append(answer_likelihood)

DF_MODEL_PREDICTIONS["answer_likelihood"] = all_answer_likelihoods
DF_MODEL_PREDICTIONS["correct_answer_likelihood"] = all_correct_answer_likelihood

#%%
DF_MODEL_PREDICTIONS.head()

#%%
DF_MODEL_PREDICTIONS["complete_match"] = [complete_match(correct_answer_text, answer)
                                          for correct_answer_text, answer 
                                          in zip(DF_MODEL_PREDICTIONS["correct_answer_text"], DF_MODEL_PREDICTIONS["answer"])]

#%%
DF_MODEL_PREDICTIONS["substring_match"] = [substring_match(correct_answer_text, answer)
                                           for correct_answer_text, answer
                                           in zip(DF_MODEL_PREDICTIONS["correct_answer_text"], DF_MODEL_PREDICTIONS["answer"])]


#%%
DF_MODEL_PREDICTIONS.head(50)

#%%
DF_MODEL_PREDICTIONS_PRED_ANSABLE = DF_MODEL_PREDICTIONS[DF_MODEL_PREDICTIONS["answer"] != ""]
DF_MODEL_PREDICTIONS_PRED_ANSABLE_IS_NONANSABLE = DF_MODEL_PREDICTIONS_PRED_ANSABLE[[row == [] for row in DF_MODEL_PREDICTIONS_PRED_ANSABLE["correct_answer_text"]]]
print("#PredAnswerable - IsNonanswerable", len(DF_MODEL_PREDICTIONS_PRED_ANSABLE_IS_NONANSABLE))
#%%
DF_MODEL_PREDICTIONS_PRED_ANSABLE_IS_ANSABLE = DF_MODEL_PREDICTIONS_PRED_ANSABLE[[row != [] for row in DF_MODEL_PREDICTIONS_PRED_ANSABLE["correct_answer_text"]]]
print("#PredAnswerable - IsAnswerable", len(DF_MODEL_PREDICTIONS_PRED_ANSABLE_IS_ANSABLE))
#%%
DF_MODEL_PREDICTIONS_PRED_NONANSABLE = DF_MODEL_PREDICTIONS[DF_MODEL_PREDICTIONS["answer"] == ""]
DF_MODEL_PREDICTIONS_PRED_NONANSABLE_IS_NONANSABLE = DF_MODEL_PREDICTIONS_PRED_NONANSABLE[[row == [] for row in DF_MODEL_PREDICTIONS_PRED_NONANSABLE["correct_answer_text"]]]
print("#PredNonanswerable - IsNonanswerable", len(DF_MODEL_PREDICTIONS_PRED_NONANSABLE_IS_NONANSABLE))
#%%
DF_MODEL_PREDICTIONS_PRED_NONANSABLE_IS_ANSABLE = DF_MODEL_PREDICTIONS_PRED_NONANSABLE[[row != [] for row in DF_MODEL_PREDICTIONS_PRED_NONANSABLE["correct_answer_text"]]]
print("#PredNonanswerable - IsAnswerable", len(DF_MODEL_PREDICTIONS_PRED_NONANSABLE_IS_ANSABLE))
#%%
DF_MODEL_PREDICTIONS_DROPNA = DF_MODEL_PREDICTIONS.dropna(subset=["answer_likelihood", "correct_answer_likelihood"])

DF_MODEL_PREDICTIONS_correct = DF_MODEL_PREDICTIONS_DROPNA[DF_MODEL_PREDICTIONS_DROPNA["substring_match"] == True]
DF_MODEL_PREDICTIONS_wrong = DF_MODEL_PREDICTIONS_DROPNA[DF_MODEL_PREDICTIONS_DROPNA["substring_match"] == False]
print("#PredAnswerable - IsAnswerable - True", len(DF_MODEL_PREDICTIONS_correct))
print("#PredAnswerable - IsAnswerable - False", len(DF_MODEL_PREDICTIONS_wrong))

#%%
len(DF_MODEL_PREDICTIONS_correct)
#%%
len(DF_MODEL_PREDICTIONS_wrong)
#%%
DF_MODEL_PREDICTIONS_DROPNA

#%%
plt.boxplot([DF_MODEL_PREDICTIONS_correct["answer_likelihood"],
             DF_MODEL_PREDICTIONS_wrong["answer_likelihood"],
             DF_MODEL_PREDICTIONS_correct["correct_answer_likelihood"],
             DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"]])


#%%
plt.scatter(DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"],
            DF_MODEL_PREDICTIONS_wrong["answer_likelihood"],
            c=DF_MODEL_PREDICTIONS_wrong["substring_match"])
plt.plot([0, 0.8], [0, 0.8], color='k', linestyle='-', linewidth=2)
plt.xlabel('correct answer likelihood')
plt.ylabel('predicted answer likelihood')
plt.savefig("scatte_wrong_answers.jpg")
plt.show()

#%%
plt.scatter(DF_MODEL_PREDICTIONS_DROPNA["correct_answer_likelihood"],
            DF_MODEL_PREDICTIONS_DROPNA["answer_likelihood"],
            c=DF_MODEL_PREDICTIONS_DROPNA["complete_match"])


#%%
plt.scatter(DF_MODEL_PREDICTIONS_DROPNA["correct_answer_likelihood"],
            DF_MODEL_PREDICTIONS_DROPNA["answer_likelihood"],
            c=DF_MODEL_PREDICTIONS_DROPNA["substring_match"])
plt.xlabel('your xlabel')
plt.ylabel('your ylabel')
plt.show()
#%%
BINS = 25
plt.hist(DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"],
         BINS,
         facecolor='blue',
         alpha=0.5)
plt.show()
plt.hist(DF_MODEL_PREDICTIONS_correct["correct_answer_likelihood"],
         BINS,
         facecolor='blue',
         alpha=0.5)
plt.show()

#%%
plt.hist(DF_MODEL_PREDICTIONS_wrong["answer_likelihood"],
         BINS,
         acecolor='blue',
         alpha=0.5)
plt.show()
plt.hist(DF_MODEL_PREDICTIONS_correct["answer_likelihood"],
         BINS,
         facecolor='blue',
         alpha=0.5)
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(DF_MODEL_PREDICTIONS_wrong["answer_likelihood"],
        BINS,
        facecolor='blue',
        alpha=0.5)
ax.hist(DF_MODEL_PREDICTIONS_wrong["correct_answer_likelihood"],
        BINS,
        facecolor='blue',
        alpha=0.5)
plt.xlabel('(correct) answer likelihood')
plt.title('Histogram of (correct) Answer Likelihood for wrongly predicted Answers')
plt.savefig('hist_wrong_answer_likelihood.png')
plt.show()

#%%
DF_MODEL_PREDICTIONS_correct.head()
DF_MODEL_PREDICTIONS_wrong.head()
