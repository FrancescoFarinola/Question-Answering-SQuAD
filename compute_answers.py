import numpy as np
from os.path import isfile
import sys
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.utils.np_utils import to_categorical

import preprocess
import load_data
import utils
import drqa_model
import bidaf_model
import our_model

from settings import MAX_CONTEXT_LENGTH, MAX_QUESTION_LENGTH, MODEL, MODELS_DIR, BATCH_SIZE, EMBEDDING_DIM

if __name__ == '__main__':
    # name of the input file
    path_to_file = str(sys.argv[1])
    try:
        assert isfile(path_to_file)
    except AssertionError:
        print(f'file {path_to_file} not found')
        sys.exit(-1)

    # name of the output file
    predictions_file = 'predictions.json'  # default name
    if len(sys.argv) >= 3:
        out_file = str(sys.argv[2])
        predictions_file = out_file

    # load data
    print("reading:", path_to_file)
    df = load_data.load_dataset_without_answer(path_to_file)

    # apply preprocessing
    print("Preprocessing data...")
    PREPROCESSING_PIPELINE1 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.spell_correction,
                               preprocess.lemmatization,
                               preprocess.lower,
                               preprocess.strip_text]
    df1 = df.copy()
    df1, tmp1 = preprocess.apply_preprocessing(df1, PREPROCESSING_PIPELINE1, text=False)

    # load already saved content
    load = (isfile(f'{MODELS_DIR}/word_listing.csv') and
            isfile(f'{MODELS_DIR}/word2idx.json') and
            isfile(f'{MODELS_DIR}/idx2word.json') and
            isfile(f'{MODELS_DIR}/tokenizer.json') and
            isfile(f'{MODELS_DIR}/embedding_matrix.csv') and
            # char embedding matrix is loaded only in case of 'our_model' or 'bidaf'
            (not(MODEL == "our_model" or MODEL == 'bidaf') or
             isfile(f"{MODELS_DIR}/char_embedding_matrix.csv")))
    try:
        assert load
    except AssertionError:
        print("Missing files")
        sys.exit(1)

    print("Loading matrices, tokenizers and dictionaries... ")
    with open(f'{MODELS_DIR}/word2idx.json') as f:
        df_word_to_idx = json.load(f)
    with open(f'{MODELS_DIR}/idx2word.json') as f:
        df_idx_to_word = json.load(f)
    with open(f'{MODELS_DIR}/tokenizer.json') as f:
        tokenizer_json = json.load(f)
        df_tokenizer = tokenizer_from_json(tokenizer_json)
    df_word_listing = np.genfromtxt(f'{MODELS_DIR}/word_listing.csv', delimiter=',', encoding='utf-8', dtype='str')
    embedding_matrix = np.genfromtxt(f'{MODELS_DIR}/embedding_matrix.csv', delimiter=',')
    df_idx_to_word = dict(zip([int(k) for k in df_idx_to_word.keys()], df_idx_to_word.values()))
    print("Done")

    # padding
    context_padded = utils.pad(df1.context, df_tokenizer, MAX_CONTEXT_LENGTH)
    question_padded = utils.pad(df1.question, df_tokenizer, MAX_QUESTION_LENGTH)

    if MODEL == 'drqa' or MODEL == "our_model":
        # compute pos, ner, em, tf
        tag2idx, idx2tag = utils.create_pos_dicts()
        ner2idx, idx2ner = utils.create_ner_dicts()
        pos_embedding_matrix = to_categorical(list(idx2tag.keys()))
        ner_embedding_matrix = to_categorical(list(idx2ner.keys()))

        em_input = utils.compute_exact_match(df1, MAX_CONTEXT_LENGTH)
        tf_input = utils.compute_tf(df1, MAX_CONTEXT_LENGTH)
        pos_input = utils.compute_pos(df1, tag2idx, MAX_CONTEXT_LENGTH)
        ner_input = utils.compute_ner(df1, ner2idx, MAX_CONTEXT_LENGTH)

        x = {'context': context_padded, 'question': question_padded, 'pos': pos_input,
             'ner': ner_input, 'em': em_input, 'tf': tf_input}

        # build model
        if MODEL == 'drqa':
            model = drqa_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                           embedding_matrix, pos_embedding_matrix, ner_embedding_matrix)
        else:
            char_embedding_matrix = np.genfromtxt(f'{MODELS_DIR}/char_embedding_matrix.csv', delimiter=',')
            model = our_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                          embedding_matrix, char_embedding_matrix, pos_embedding_matrix,
                                          ner_embedding_matrix)
    else:
        x = {'context': context_padded, 'question': question_padded}

        # build model
        if MODEL == 'bidaf':
            char_embedding_matrix = np.genfromtxt(f'{MODELS_DIR}/char_embedding_matrix.csv', delimiter=',')
            model = bidaf_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                            embedding_matrix, char_embedding_matrix)

    # load model weights
    model.load_weights(f"{MODELS_DIR}/{MODEL}_weights.h5")

    # compute and save predictions
    predictions = utils.computing_predictions(model, df, x, BATCH_SIZE)
    print("Saving predictions as json...")
    with open(predictions_file, 'w') as outfile:
        json.dump(predictions, outfile)
