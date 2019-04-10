import pandas as pd
import ast
import train

TRAIN_FILENAME = "data/processed/train.csv"
DEV_FILENAME = "data/processed/dev.csv"

# Make up some training data
df_askable_paragraph = pd.read_csv(TRAIN_FILENAME)
df_askable_paragraph["paragraph_context_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["paragraph_context_tokens"]]
df_askable_paragraph["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["askable_tokens"]]
training_data = list(zip(list(df_askable_paragraph["paragraph_context_tokens"]), list(df_askable_paragraph["askable_tokens"])))

df_askable_paragraph = pd.read_csv(DEV_FILENAME)
df_askable_paragraph["paragraph_context_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["paragraph_context_tokens"]]
df_askable_paragraph["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["askable_tokens"]]
dev_data = list(zip(list(df_askable_paragraph["paragraph_context_tokens"]), list(df_askable_paragraph["askable_tokens"])))

train(training_data, dev_data, embedding_dim, hidden_dim, no_epochs)
