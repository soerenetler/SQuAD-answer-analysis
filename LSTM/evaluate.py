from utils import prepare_sequence
from sklearn.metrics import accuracy_score, precision_score, f1_score

def evaluate(data, model, word_to_ix, tag_to_ix):
    conc_ys = []
    conc_y_pred = []

    for tokens, askable_tags in data:
        precheck_sent = prepare_sequence(tokens, word_to_ix)
        y = [tag_to_ix[w] for w in askable_tags]
        y_pred = model(precheck_sent)[1]

        conc_ys += y
        conc_y_pred += y_pred

    train_accuracy = accuracy_score(conc_ys, conc_y_pred)
    train_precision = precision_score(conc_ys, conc_y_pred, average="macro")
    train_f1 = f1_score(conc_ys, conc_y_pred, average="macro")

    return train_accuracy, train_precision, train_f1
