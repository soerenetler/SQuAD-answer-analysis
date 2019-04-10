import pandas as pd
import ast
import train

TRAIN_FILENAME = "../data/preprocessedData/train_IO_with_plausible_answers.csv"
DEV_FILENAME   = "../data/preprocessedData/train_IO_with_plausible_answers_test.csv"
EMBED_FILENAME = "../data/glove/glove.6B.300d.txt"

# Make up some training data
df_askable_paragraph = pd.read_csv(TRAIN_FILENAME)
df_askable_paragraph["paragraph_context_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["paragraph_context_tokens"]]
df_askable_paragraph["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["askable_tokens"]]
training_data = list(zip(list(df_askable_paragraph["paragraph_context_tokens"]), list(df_askable_paragraph["askable_tokens"])))

df_askable_paragraph = pd.read_csv(DEV_FILENAME)
df_askable_paragraph["paragraph_context_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["paragraph_context_tokens"]]
df_askable_paragraph["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["askable_tokens"]]
dev_data = list(zip(list(df_askable_paragraph["paragraph_context_tokens"]), list(df_askable_paragraph["askable_tokens"])))


def load_glove(path):
    word2vec = {}
    with open(path) as lines:
        for line in lines:
            word, *arr = line.split()
            word2vec[word] = np.array(list(map(float, arr)))
    return word2vec


word2vec = load_glove(EMBED_FILENAME)

losses = train(training_data, dev_data, word2vec, embedding_dim, hidden_dim, no_epochs)
