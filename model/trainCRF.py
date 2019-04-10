#%%
import ast
import itertools
import pickle
from tqdm import tqdm
import spacy
import pandas as pd

from sklearn.metrics import make_scorer, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite

from model.crf_utils import text2features, visualize_rs_result

#%%
nlp = spacy.load('en')


#%%
TRAIN_FILENAME = "data/preprocessedData/train_IO_with_plausible_answers.csv"
TEST_FILENAME = "data/preprocessedData/dev_IO_with_plausible_answers.csv"

df_askable_paragraph_train = pd.read_csv(TRAIN_FILENAME)
df_askable_paragraph_train["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_train["askable_tokens"]]

#%%
df_askable_paragraph_train_sample = df_askable_paragraph_train.sample(n=1000, random_state=1)


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
X_train_sample = [text2features(nlp(s), pos_features=POS_FEATURES, ent_type_features=ENT_TYPE_FEATURES, lemma_features=LEMMA_FEATURES, is_features=IS_FEATURES, position_features=POSITION_FEATURES, bias=BIAS, begin=BEGIN, end=END) for s in tqdm(df_askable_paragraph_train_sample["paragraph_context"])]
y_train_sample = list(df_askable_paragraph_train_sample["askable_tokens"])

#%%
class Custom_CRF(sklearn_crfsuite.CRF):
    def predict_proba(self, X):
        return self.predict_marginals(X)


#%%
def custom_roc_auc_score(y_trues, prob_pred):
    y_true_merged = [y_true == 'I' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I'] for y_pred in list(itertools.chain(*prob_pred))]

    return roc_auc_score(y_true_merged, y_pred_list)

def custom_log_loss(y_trues, prob_pred):
    y_true_merged = [y_true == 'I' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I'] for y_pred in list(itertools.chain(*prob_pred))]
    
    return log_loss(y_true_merged, y_pred_list)


#%%
crf = Custom_CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True,
    min_freq=5
)
params_space = {
    'c1': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], #scipy.stats.expon(scale=5),
    'c2': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000] #scipy.stats.expon(scale=50)
}

f1_scorer = make_scorer(custom_roc_auc_score, needs_proba=True) #, greater_is_better=False)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=10,
                        n_jobs=-1,
                        n_iter=99,
                        scoring=f1_scorer)
rs.fit(X_train_sample, y_train_sample)

#%%
visualize_rs_result(rs)

#%%
#X_train = [text2features(nlp(s), pos_features=POS_FEATURES, ent_type_features=ENT_TYPE_FEATURES, lemma_features=LEMMA_FEATURES, is_features=IS_FEATURES, position_features=POSITION_FEATURES, bias=BIAS, begin=BEGIN, end=END) for s in tqdm(df_askable_paragraph_train["paragraph_context"])]
#y_train = list(df_askable_paragraph_train["askable_tokens"])

#%%
crf = rs.best_estimator_
crf.fit(X_train_sample, y_train_sample)

#%%
filename = "crf_model_sample1000.obj"
pickle.dump(crf, open(filename, 'wb'))