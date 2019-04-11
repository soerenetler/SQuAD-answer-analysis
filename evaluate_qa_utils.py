"""Usefull functions for the evaluation of qa model"""
import re
import spacy
from model.crf_utils import text2features


POS_FEATURES = True
ENT_TYPE_FEATURES = True
LEMMA_FEATURES = True
IS_FEATURES = True
POSITION_FEATURES = True
BIAS = True
BEGIN = -1
END = 1

NLP = spacy.load('en')

def char_index_2_token_index(paragraph, substring_start, substring_text):
    """convert char indicies into token indicies

    Keyword arguments:
    paragraph -- paragraph as a string
    substring_start -- start char index of the tokens
    substring_text -- substring
    """
    doc = NLP(paragraph)
    span = doc.char_span(substring_start, substring_start+len(substring_text))

    try:
        return span.start, span.end
    except AttributeError:
        return (None, None)

def get_likelihoods(paragraph_text, correct_answer_token_indices, answer_token_indices, crf):
    """get the predicted likelihod of the correct answer tokens and
    the predicted answer tokens by the crf model.

    Keyword arguments:
    paragraph_text -- paragraph as a string
    correct_answer_token_indices -- indicies of the correct answers tokens (start, end)
    answer_token_indices -- indicies of the predicted answers tokens (start, end)
    crf -- CRF model to use for the predictions
    """
    paragraph = NLP(paragraph_text)
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
        correct_answer_likelihoods.append(
            max(predictions[correct_answer_token_start:correct_answer_token_end]))

    answer_likelihoods = []
    for answer_token_index in set(answer_token_indices):
        answer_token_start, answer_token_end = answer_token_index
        answer_likelihoods.append(max(predictions[answer_token_start:answer_token_end]))

    return max(correct_answer_likelihoods, default=None), max(answer_likelihoods, default=None)


def string_to_token_index(paragraph, substring):
    """find a string in a text and return the list of token idecies (start, end)
    If substring is not in the paragraph an empty list is returned

    Keyword arguments:
    paragraph -- paragraph as a string
    substring -- substring to find the indedies of
    """
    if substring == "":
        return []
    else:
        result = []
        for match in re.finditer(re.escape(substring), paragraph):
            index = char_index_2_token_index(paragraph, match.start(), substring)
            if not index == (None, None):
                result.append(index)
        return result

def find_property_to_questionid(question_id, property_name, df_questions):
    """find a property in the DF_QUSETIONS for a specific question ID

    Keyword arguments:
    question_id -- question ID string
    property_name -- name of the df column of the property to find
    DF_QUESTIONS -- dataframe with the questions
    """
    try:
        return df_questions.loc[question_id][property_name]
    except KeyError:
        return None

def substring_match(correct_answer_texts, answer_text):
    """Returns true if the answer text is a substring
       of one of the strings in the array of correct answers
       (or vice verza)

    Keyword arguments:
    correct_answer_texts -- list of correct answer strings
    answer_text -- predicted answer string
    """
    if answer_text == "":
        return False
    return any((answer_text in correct_answer_text) or (correct_answer_text in answer_text) for correct_answer_text in correct_answer_texts)

def complete_match(correct_answer_texts, answer_text):
    """Returns true if the answer text is identical
       to one of the strings in the array of correct answers

    Keyword arguments:
    correct_answer_texts -- list of correct answer strings
    answer_text -- predicted answer string
    """
    if answer_text == "":
        return False
    return answer_text in correct_answer_texts
