#%%
import ast
import pickle

import spacy
import pandas as pd
import eli5
from sklearn_crfsuite import metrics
from tqdm import tqdm

from model.crf_utils import text2features, Custom_CRF
from model.evaluation_utils import visualize_transitions, print_annotated_text, crf_roc_curve, crf_log_loss

#%%
POS_FEATURES = True
ENT_TYPE_FEATURES = True
LEMMA_FEATURES = True
IS_FEATURES = True
POSITION_FEATURES = True
BIAS = True
BEGIN = -1
END = 1

#%%
filename = "model/trainedModels/crf_sample_1000.obj"
with open(filename, 'rb') as f:
    crf = pickle.load(f)

#%%
TEST_FILENAME = "data/preprocessedData/dev_IO_with_plausible_answers.csv"
df_askable_paragraph_test = pd.read_csv(TEST_FILENAME)
df_askable_paragraph_test["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_test["askable_tokens"]]

#%%
nlp = spacy.load('en')

#%%
X_test = [text2features(nlp(s), pos_features=POS_FEATURES, ent_type_features=ENT_TYPE_FEATURES, lemma_features=LEMMA_FEATURES, is_features=IS_FEATURES, position_features=POSITION_FEATURES, bias=BIAS, begin=BEGIN, end=END) for s in tqdm(df_askable_paragraph_test["paragraph_context"])]
y_test = list(df_askable_paragraph_test["askable_tokens"])


#%%
# Evaluation on Test data
y_pred_test = crf.predict(X_test)
y_pred_test_marginals = crf.predict_marginals(X_test)
#%%
print(metrics.flat_classification_report(
    y_test, y_pred_test, digits=5
))

#%%
visualize_transitions(crf)

#%%
paragraph = nlp("A Japan-exclusive manga series based on Twilight Princess, penned and illustrated by Akira Himekawa, was first released on February 8, 2016. The series is available solely via publisher Shogakukan's MangaOne mobile application. While the manga adaptation began almost ten years after the initial release of the game on which it is based, it launched only a month before the release of the high-definition remake.")
y_true = ['O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

tokens =[token.text for token in paragraph]
y_pred = crf.predict_marginals_single(text2features(paragraph, pos_features=POS_FEATURES, ent_type_features=ENT_TYPE_FEATURES, lemma_features=LEMMA_FEATURES, is_features=IS_FEATURES, position_features=POSITION_FEATURES, bias=BIAS, begin=BEGIN, end=END))

#%%
print_annotated_text(tokens, y_pred, y_trues=y_true)

#%%
for index, row in df_askable_paragraph_test.iterrows():
    paragraph = nlp(row["paragraph_context"])
    y_true = row["askable_tokens"]
    tokens =[token.text for token in paragraph]
    y_pred = crf.predict_marginals_single(text2features(paragraph, pos_features=POS_FEATURES, ent_type_features=ENT_TYPE_FEATURES, lemma_features=LEMMA_FEATURES, is_features=IS_FEATURES, position_features=POSITION_FEATURES, bias=BIAS, begin=BEGIN, end=END))

    print_annotated_text(tokens, y_pred, y_trues=y_true)

#%%
print("Log Loss on the test data: " + str(crf_log_loss(y_test, y_pred_test)))

#%%
crf_roc_curve(y_test, y_pred_test)

#%%
eli5.show_weights(crf, top=25)
