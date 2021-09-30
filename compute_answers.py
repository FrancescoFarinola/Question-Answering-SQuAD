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
    #MAX_CONTEXT_LENGTH = settings.MAX_CONTEXT_LENGTH
    #MAX_TEXT_LENGTH = settings.MAX_TEXT_LENGTH
    #MAX_QUESTION_LENGTH = settings
    #MODEL = "our_model"
    #read data
    path_to_file = str(sys.argv[1])
    assert isfile(path_to_file)
    print("reading:", path_to_file)
    df = load_data.load_dataset_without_answer(path_to_file)
    #apply preprocessing
    PREPROCESSING_PIPELINE1 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.spell_correction,
                               preprocess.lemmatization,
                               preprocess.lower,
                               preprocess.strip_text]

    print("Preprocessing data...")
    df1 = df.copy()
    df1, tmp1 = preprocess.apply_preprocessing(df1, PREPROCESSING_PIPELINE1, text=False)

    # load already saved content or compute it from scratch
    load = (isfile(f'{MODELS_DIR}/word_listing.csv') and
            isfile(f'{MODELS_DIR}/word2idx.json') and
            isfile(f'{MODELS_DIR}/idx2word.json') and
            isfile(f'{MODELS_DIR}/tokenizer.json') and
            isfile(f'{MODELS_DIR}/embedding_matrix.csv'))
    print("load:", load)

    try:
        assert load
        print("Loading matrices, tokenizers and dictionaries... ")
        # load pre-saved
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
    except:
        print("File are missing")

    context_padded = utils.pad(df1.context, df_tokenizer, MAX_CONTEXT_LENGTH)
    question_padded = utils.pad(df1.question, df_tokenizer, MAX_QUESTION_LENGTH)

    if MODEL == 'drqa' or MODEL == "our_model":
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
        #y = {'start': s_one, 'end': e_one}


        if MODEL == 'drqa':
            model = drqa_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                           embedding_matrix, pos_embedding_matrix, ner_embedding_matrix)
        else:
            assert isfile(f"{MODELS_DIR}/char_embedding_matrix.csv")
            char_embedding_matrix = np.genfromtxt(f'{MODELS_DIR}/char_embedding_matrix.csv', delimiter=',')
            model = our_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                          embedding_matrix, char_embedding_matrix, pos_embedding_matrix,
                                          ner_embedding_matrix)
    else:
        x = {'context': context_padded, 'question': question_padded}
        #y = {'start': s_one, 'end': e_one}

        if MODEL == 'bidaf':
            assert isfile(f"{MODELS_DIR}/char_embedding_matrix.csv")
            char_embedding_matrix = np.genfromtxt(f'{MODELS_DIR}/char_embedding_matrix.csv', delimiter=',')
            model = bidaf_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                            embedding_matrix, char_embedding_matrix)

    model.load_weights(f"{MODELS_DIR}/{MODEL}_weights.h5")

    #print("Evalutating model...")
    #evaluation = model.evaluate(x, y, batch_size=utils.BATCH_SIZE)
    #print(evaluation)

    predictions = utils.computing_predictions(model, df, x, BATCH_SIZE)
    print("Saving predictions as json...")
    with open('predictions.json', 'w') as outfile: # va a sovrascrivere altre predicition??
        json.dump(predictions, outfile)

    #f1, precision, recall = utils.evaluate_model(model, MAX_CONTEXT_LENGTH, df1, x)
    #print(f"F1: {f1}\t Precision: {precision}\t Recall: {recall}\t")




