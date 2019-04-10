import json
import random
import pandas as pd
import spacy
from tqdm import tqdm

def get_answer_sentence(nlp_context, answer_start, answer_text):
    answer_span = nlp_context.char_span(answer_start, answer_start+len(answer_text))
    # TODO There should be a better way to handle this
    if answer_span is None:
        global INVALID_SPAN_DF
        INVALID_SPAN_DF = INVALID_SPAN_DF.append({"paragraph":nlp_context.text,
                                                  "span_start":answer_start,
                                                  "span_end": answer_start+len(answer_text),
                                                  "span_text": answer_text},
                                                 ignore_index=True)
    elif answer_span.text != answer_text:
        print("answer_span.text != answer_text")
        print("answer span: {}, answer text: {}".format(answer_span.text, answer_text))
    else:
        return answer_span.sent.text, answer_span.start, answer_span.end

def read_to_dataframe(filename, labeling, include_plausible_answers=False):
    (nlp) = spacy.load('en')
    df_askable_paragraph = pd.DataFrame(columns=["text_title",
                                                 "paragraph_context",
                                                 "paragraph_context_tokens",
                                                 "askable_tokens"])

    df_question_difficulty = pd.DataFrame(columns=["paragraph_context",
                                                   "question_id",
                                                   "question_text",
                                                   "correct_answer"])

    with open(filename) as json_data:
        json_dict = json.load(json_data)
        for text in tqdm(json_dict['data']):
            text_title = text['title']
            for paragraph in text['paragraphs']:
                paragraph_context = paragraph['context']
                nlp_context = nlp(paragraph_context)
                paragraph_context_tokens = [t.text for t in nlp_context]
                askable_tokens = ["O"]*len(nlp_context)
                for question in paragraph['qas']:
                    question_id = question["id"]
                    question_text = question["question"]
                    correct_answer_texts = []
                    correct_answer_char_index = []
                    correct_answer_token_index = []
                    plausible_answer_texts = []
                    plausible_answer_char_index = []
                    plausible_answer_token_index = []

                    if question['answers']:
                        for answer in question['answers']:
                            correct_answer_texts.append(answer['text'])
                            correct_answer_char_index.append((answer['answer_start'], answer['answer_start'] + len(answer['text'])))

                            result = get_answer_sentence(nlp_context,
                                                         answer['answer_start'],
                                                         answer['text'])
                            if result is None:
                                continue
                            answer_sentence, answer_span_start, answer_span_end = result
                            correct_answer_token_index.append((answer_span_start, answer_span_end))
                            if labeling == "IOB":
                                if (answer_span_end-answer_span_start) == 1:
                                    askable_tokens[answer_span_start] = "I"
                                else:
                                    askable_tokens[answer_span_start] = "B"
                                    askable_tokens[answer_span_start+1:answer_span_end] = ["I"]*(answer_span_end-answer_span_start-1)
                            elif labeling == "IO":
                                askable_tokens[answer_span_start:answer_span_end] = ["I"]*(answer_span_end-answer_span_start)
                            else:
                                raise ValueError("Currently only IOB and IO labeling is supported")

                    if "plausible_answers" in question.keys() and question["plausible_answers"]:
                        for plausible_answer in question["plausible_answers"]:
                            plausible_answer_texts.append(plausible_answer['text'])
                            plausible_answer_char_index.append((plausible_answer['answer_start'], plausible_answer['answer_start'] + len(plausible_answer['text'])))

                            result = get_answer_sentence(nlp_context,
                                                          plausible_answer['answer_start'],
                                                          plausible_answer['text'])
                            if result is None:
                                continue
                            answer_sentence, answer_span_start, answer_span_end = result
                            plausible_answer_token_index.append((answer_span_start,answer_span_end))
                            if labeling == "IOB":
                                if (answer_span_end-answer_span_start) == 1:
                                    askable_tokens[answer_span_start] = "I"
                                else:
                                    askable_tokens[answer_span_start] = "B"
                                    askable_tokens[answer_span_start+1:answer_span_end] = ["I"]*(answer_span_end-answer_span_start-1)
                            elif labeling == "IO":
                                askable_tokens[answer_span_start:answer_span_end] = ["I"]*(answer_span_end-answer_span_start)
                            else:
                                raise ValueError("Currently only IOB and IO labeling is supported")
                
                
                    df_question_difficulty = df_question_difficulty.append({"paragraph_context": paragraph_context,
                                                                            "correct_answer_text": correct_answer_texts,
                                                                            "correct_answer_char_index": correct_answer_char_index,
                                                                            "correct_answer_token_index": correct_answer_token_index,
                                                                            "plausible_answer_text": plausible_answer_texts,
                                                                            "plausible_answer_char_index": plausible_answer_char_index,
                                                                            "plausible_answer_token_index": plausible_answer_token_index,
                                                                            "question_id":question_id,
                                                                            "question_text":question_text,
                                                                            'paragraph_context_tokens':paragraph_context_tokens
                                                                            }, ignore_index=True)

                df_askable_paragraph = df_askable_paragraph.append({'text_title':text_title,
                                                                    'paragraph_context':paragraph_context,
                                                                    'paragraph_context_tokens':paragraph_context_tokens,
                                                                    'askable_tokens': askable_tokens},
                                                                    ignore_index=True)

    return df_askable_paragraph, df_question_difficulty

def create_train_dev_test_random(train_filename, dev_filename, labeling):
    df1_askable_paragraph = read_to_dataframe(train_filename, labeling)
    df2_askable_paragraph = read_to_dataframe(dev_filename, labeling)

    total_df_askable_paragraph = pd.concat(
        [df1_askable_paragraph, df2_askable_paragraph], ignore_index=True)

    textnames = list(total_df_askable_paragraph["text_title"].unique())

    textnames.sort()
    random.seed(1)
    random.shuffle(textnames)

    split_1 = int(0.8 * len(textnames))
    split_2 = int(0.9 * len(textnames))

    train_textnames = textnames[:split_1]
    dev_textnames = textnames[split_1:split_2]
    test_textnames = textnames[split_2:]

    train_df_askable_paragraph = total_df_askable_paragraph[
        [textname in train_textnames for textname in total_df_askable_paragraph["text_title"]]]
    dev_df_askable_paragraph = total_df_askable_paragraph[
        [textname in dev_textnames for textname in total_df_askable_paragraph["text_title"]]]
    test_df_askable_paragraph = total_df_askable_paragraph[
        [textname in test_textnames for textname in total_df_askable_paragraph["text_title"]]]

    train_df_askable_paragraph.to_csv("train_"+labeling+".csv")
    dev_df_askable_paragraph.to_csv("dev_"+labeling+".csv")
    test_df_askable_paragraph.to_csv("test_"+labeling+".csv")

    INVALID_SPAN_DF.to_csv("invalid_spans.csv")

def create_train_dev_test(train_filename, dev_filename, labeling):
    df1_askable_paragraph, df1_questions = read_to_dataframe(train_filename, labeling, include_plausible_answers=True)
    df2_askable_paragraph, df2_questions = read_to_dataframe(dev_filename, labeling, include_plausible_answers=True)
    df1_askable_paragraph.to_csv("data/preprocessedData/train_"+labeling+"_with_plausible_answers.csv")
    df2_askable_paragraph.to_csv("data/preprocessedData/dev_"+labeling+"_with_plausible_answers.csv")
    df1_questions.to_csv("data/preprocessedData/train_questions.csv")
    df2_questions.to_csv("data/preprocessedData/dev_questions.csv")



TRAIN_FILENAME = 'data/rawData/train-v2.0.json'
DEV_FILENAME = 'data/rawData/dev-v2.0.json'
LABELING = "IO"

INVALID_SPAN_DF = pd.DataFrame(columns=["paragraph",
                                        "span_start",
                                        "span_end",
                                        "span_text"])

create_train_dev_test(TRAIN_FILENAME, DEV_FILENAME, LABELING)