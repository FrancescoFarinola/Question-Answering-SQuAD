import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


def load_dataset():
    path = './data'
    filename = "training_set.json"

    with open(f"{path}/{filename}") as f:
        data = json.load(f)

    dataframe_rows = []
    for d in data["data"]:
        title = d["title"]
        paragraphs = d["paragraphs"]
        for p in paragraphs:
            context = p["context"]
            qas = p["qas"]
            for q in qas:
                answers = q["answers"]
                question = q["question"]
                qid = q["id"]
                for a in answers:
                    answer_start = a["answer_start"]
                    text = a["text"]

                    dataframe_row = {
                        "title": title,
                        "context": context,
                        "answer_start": answer_start,
                        "text": text,
                        "question": question,
                        "id": qid
                    }
                    dataframe_rows.append(dataframe_row)

    return pd.DataFrame(dataframe_rows)


def remove_error_rows(dataframe, path, filename):
    with open(f"{path}/{filename}", encoding='utf-8') as f_errors:
        errors = f_errors.read().splitlines()
    dataframe = dataframe[dataframe['id'].isin(errors)]
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def remove_2occ_rows(dataframe):
    tmpocc = dataframe.apply(lambda x: x.context.count(x.text), axis=1)
    tmpindices = np.where(tmpocc > 1)
    dataframe = dataframe.loc[tmpindices]
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def split_test_set(dataframe):
    ts_df1 = remove_error_rows(dataframe, path="./data", filename="error IDs.txt")
    ts_df2 = remove_2occ_rows(dataframe)
    ts_df = pd.concat([ts_df1, ts_df2])
    ts_df.reset_index(inplace=True, drop=True)
    dataframe = dataframe[~dataframe['id'].isin(ts_df.id)]
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe, ts_df



def split_validation_set(dataframe, rate):
    tr_title, val_title = train_test_split(np.unique(dataframe.title), test_size=rate, random_state=0)
    tr_idx = np.isin(dataframe.title, tr_title)
    val_idx = np.isin(dataframe.title, val_title)

    tr_df = dataframe.loc[tr_idx]
    tr_df.reset_index(inplace=True, drop=True)
    val_df = dataframe.loc[val_idx]
    val_df.reset_index(inplace=True, drop=True)
    return tr_df, val_df


