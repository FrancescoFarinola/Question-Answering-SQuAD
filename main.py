from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.np_utils import to_categorical
import load_data
import preprocess
import basemodel
import pandas as pd
import utils
import drqa_model
import bidaf_model
import our_model
import sys
import io
import json


EMBEDDING_DIM = 100
UNITS = int(EMBEDDING_DIM/2)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataframe = load_data.load_dataset()
    print(dataframe.shape)

    train_df, test_df = load_data.split_test_set(dataframe)
    print(train_df.shape, test_df.shape)

    train_df, val_df = load_data.split_validation_set(train_df, rate=0.2)
    print(train_df.shape, val_df.shape)

    PREPROCESSING_PIPELINE1 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.spell_correction,
                               preprocess.lemmatization,
                               preprocess.lower,
                               preprocess.strip_text]

    print("Preprocessing train data...")
    train_df1 = train_df.copy()
    train_df1, train_tmp1 = preprocess.apply_preprocessing(train_df1, PREPROCESSING_PIPELINE1)

    print("Preprocessing validation data...")
    val_df1 = val_df.copy()
    val_df1, val_tmp1 = preprocess.apply_preprocessing(val_df1, PREPROCESSING_PIPELINE1)

    embedding_matrix, df_word_listing, df_tokenizer, df_word_to_idx, df_idx_to_word = utils.get_embedding_matrix(train_df1, EMBEDDING_DIM)

    #save tokenizer
    tokenizer_json = df_tokenizer.to_json()
    with io.open('./models/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    MAX_CONTEXT_LENGTH, MAX_TEXT_LENGTH, MAX_QUESTION_LENGTH = utils.get_max_length(train_df1)

    print("Padding data...")
    tr_context_padded = utils.pad(train_df1.context, df_tokenizer, MAX_CONTEXT_LENGTH)
    tr_answer_padded = utils.pad(train_df1.text, df_tokenizer, MAX_TEXT_LENGTH)
    tr_question_padded = utils.pad(train_df1.question, df_tokenizer, MAX_QUESTION_LENGTH)

    val_context_padded = utils.pad(val_df1.context, df_tokenizer, MAX_CONTEXT_LENGTH)
    val_answer_padded = utils.pad(val_df1.text, df_tokenizer, MAX_TEXT_LENGTH)
    val_question_padded = utils.pad(val_df1.question, df_tokenizer, MAX_QUESTION_LENGTH)

    print("Computing start and end indexes...")
    train_df1['s_idx'] = train_df.apply(
        lambda x: len(preprocess.preprocessing(x.context[:x.answer_start], PREPROCESSING_PIPELINE1).split()), axis=1)
    train_df1['e_idx'] = train_df1.apply(lambda x: x.s_idx + len(x.text.split()) - 1, axis=1)

    val_df1['s_idx'] = val_df.apply(
        lambda x: len(preprocess.preprocessing(x.context[:x.answer_start], PREPROCESSING_PIPELINE1).split()), axis=1)
    val_df1['e_idx'] = val_df1.apply(lambda x: x.s_idx + len(x.text.split()) - 1, axis=1)

    model_name = 'our_model'

    if model_name == 'basemodel' or model_name == None:
        pass

    elif model_name == 'drqa':

        pos_listing = ["$", "``", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT",
                       "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP",
                       "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH",
                       "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX", "_SP"]

        ner_listing = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                       "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

        train_em_input = utils.compute_exact_match(train_df1, MAX_CONTEXT_LENGTH)
        train_tf_input = utils.compute_tf(train_df1, MAX_CONTEXT_LENGTH)
        val_em_input = utils.compute_exact_match(val_df1, MAX_CONTEXT_LENGTH)
        val_tf_input = utils.compute_tf(val_df1, MAX_CONTEXT_LENGTH, train_set=False)

        tag2idx, idx2tag = utils.create_pos_dicts(pos_listing)
        ner2idx, idx2ner = utils.create_ner_dicts(ner_listing)

        train_pos_input = utils.compute_pos(train_df1, tag2idx, MAX_CONTEXT_LENGTH)
        val_pos_input = utils.compute_pos(val_df1, tag2idx, MAX_CONTEXT_LENGTH)
        pos_embedding_matrix = to_categorical(list(idx2tag.keys()))

        train_ner_input = utils.compute_ner(train_df1, ner2idx, MAX_CONTEXT_LENGTH)
        val_ner_input = utils.compute_ner(val_df1, ner2idx, MAX_CONTEXT_LENGTH)
        ner_embedding_matrix = to_categorical(list(idx2ner.keys()))


        model = drqa_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM, embedding_matrix,
                    pos_embedding_matrix, ner_embedding_matrix)

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics='accuracy')
        model.summary()
        plot_model(model, rankdir='TB', show_shapes=True, show_dtype=True, to_file="/models/drqa.png")

        from tensorflow import one_hot
        tr_s_one = one_hot(train_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
        tr_e_one = one_hot(train_df1.e_idx, depth=MAX_CONTEXT_LENGTH)
        val_s_one = one_hot(val_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
        val_e_one = one_hot(val_df1.e_idx, depth=MAX_CONTEXT_LENGTH)

        x_tr = {'context': tr_context_padded, 'question': tr_question_padded, 'pos': train_pos_input,
                'ner': train_ner_input, 'em': train_em_input, 'tf': train_tf_input}
        x_val = {'context': val_context_padded, 'question': val_question_padded, 'pos': val_pos_input,
                 'ner': val_ner_input, 'em': val_em_input, 'tf': val_tf_input}

        y_tr = {'start': tr_s_one, 'end': tr_e_one}
        y_val = {'start': val_s_one, 'end': val_e_one}

        mycb = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=10, batch_size=16, callbacks=[mycb])
        model.save('./models/drqa')

    elif model_name == "bidaf":

        char_embedding_matrix = utils.get_char_embeddings(df_word_listing, df_word_to_idx)

        model = bidaf_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM, embedding_matrix, char_embedding_matrix)

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics='accuracy')
        model.summary()
        plot_model(model, rankdir='TB', show_shapes=True, show_dtype=True, to_file="./models/bidaf.png")

        from tensorflow import one_hot

        tr_s_one = one_hot(train_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
        tr_e_one = one_hot(train_df1.e_idx, depth=MAX_CONTEXT_LENGTH)
        val_s_one = one_hot(val_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
        val_e_one = one_hot(val_df1.e_idx, depth=MAX_CONTEXT_LENGTH)

        x_tr = {'context': tr_context_padded, 'question': tr_question_padded}
        x_val = {'context': val_context_padded, 'question': val_question_padded}

        y_tr = {'start': tr_s_one, 'end': tr_e_one}
        y_val = {'start': val_s_one, 'end': val_e_one}

        mycb = EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=1, batch_size=16, callbacks=[mycb])
        model.save('./models/bidaf')

    elif model_name == "our_model":

        pos_listing = ["$", "``", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT",
                       "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP",
                       "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH",
                       "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX", "_SP"]

        ner_listing = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                       "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

        train_em_input = utils.compute_exact_match(train_df1, MAX_CONTEXT_LENGTH)
        train_tf_input = utils.compute_tf(train_df1, MAX_CONTEXT_LENGTH)
        val_em_input = utils.compute_exact_match(val_df1, MAX_CONTEXT_LENGTH)
        val_tf_input = utils.compute_tf(val_df1, MAX_CONTEXT_LENGTH, train_set=False)

        tag2idx, idx2tag = utils.create_pos_dicts(pos_listing)
        ner2idx, idx2ner = utils.create_ner_dicts(ner_listing)

        train_pos_input = utils.compute_pos(train_df1, tag2idx, MAX_CONTEXT_LENGTH)
        val_pos_input = utils.compute_pos(val_df1, tag2idx, MAX_CONTEXT_LENGTH)
        pos_embedding_matrix = to_categorical(list(idx2tag.keys()))

        train_ner_input = utils.compute_ner(train_df1, ner2idx, MAX_CONTEXT_LENGTH)
        val_ner_input = utils.compute_ner(val_df1, ner2idx, MAX_CONTEXT_LENGTH)
        ner_embedding_matrix = to_categorical(list(idx2ner.keys()))

        char_embedding_matrix = utils.get_char_embeddings(df_word_listing, df_word_to_idx)

        model = our_model.build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM,
                                      embedding_matrix, char_embedding_matrix, pos_embedding_matrix, ner_embedding_matrix)

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics='accuracy')
        model.summary()
        plot_model(model, rankdir='TB', show_shapes=True, show_dtype=True, to_file="./models/our_model.png")

        from tensorflow import one_hot

        tr_s_one = one_hot(train_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
        tr_e_one = one_hot(train_df1.e_idx, depth=MAX_CONTEXT_LENGTH)
        val_s_one = one_hot(val_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
        val_e_one = one_hot(val_df1.e_idx, depth=MAX_CONTEXT_LENGTH)

        x_tr = {'context': tr_context_padded, 'question': tr_question_padded, 'pos': train_pos_input,
                'ner': train_ner_input, 'em': train_em_input, 'tf': train_tf_input}
        x_val = {'context': val_context_padded, 'question': val_question_padded, 'pos': val_pos_input,
                 'ner': val_ner_input, 'em': val_em_input, 'tf': val_tf_input}

        y_tr = {'start': tr_s_one, 'end': tr_e_one}
        y_val = {'start': val_s_one, 'end': val_e_one}

        mycb = EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=1, batch_size=16, callbacks=[mycb])
        model.save('./models/our_model')

    test_df1 = test_df.copy()
    test_df1, test_tmp1 = preprocess.apply_preprocessing(test_df1, PREPROCESSING_PIPELINE1)

    ts_context_padded = utils.pad(test_df1.context, df_tokenizer, MAX_CONTEXT_LENGTH)
    ts_answer_padded = utils.pad(test_df1.text, df_tokenizer, MAX_TEXT_LENGTH)
    ts_question_padded = utils.pad(test_df1.question, df_tokenizer, MAX_QUESTION_LENGTH)

    test_df1['s_idx'] = test_df.apply(
        lambda x: len(preprocess.preprocessing(x.context[:x.answer_start], PREPROCESSING_PIPELINE1).split()), axis=1)
    test_df1['e_idx'] = test_df1.apply(lambda x: x.s_idx + len(x.text.split()) - 1, axis=1)

    ts_s_one = one_hot(test_df1.s_idx, depth=MAX_CONTEXT_LENGTH)
    ts_e_one = one_hot(test_df1.e_idx, depth=MAX_CONTEXT_LENGTH)

    if model_name == 'drqa' or model_name == "our_model":

        ts_em_input = utils.compute_exact_match(test_df1, MAX_CONTEXT_LENGTH)
        ts_tf_input = utils.compute_tf(test_df1, MAX_CONTEXT_LENGTH, train_set=False)
        ts_pos_input = utils.compute_pos(test_df1, tag2idx, MAX_CONTEXT_LENGTH)
        ts_ner_input = utils.compute_ner(test_df1, ner2idx, MAX_CONTEXT_LENGTH)

        x_ts = {'context': ts_context_padded, 'question': ts_question_padded, 'pos': ts_pos_input,
                'ner': ts_ner_input, 'em': ts_em_input, 'tf': ts_tf_input}
        y_ts = {'start': ts_s_one, 'end': ts_e_one}
    else:
        x_ts = {'context': ts_context_padded, 'question': ts_question_padded}
        y_ts = {'start': ts_s_one, 'end': ts_e_one}


    print("Evalutating model...")
    evaluation = model.evaluate(x_ts, y_ts, batch_size=16)
    print(evaluation)

    predictions = utils.computing_predictions(model, train_df, val_df, test_df, x_tr, x_val, x_ts)

    print("Saving predictions as json...")
    with open('predictions.json', 'w') as outfile:
        json.dump(predictions, outfile)

    f1, precision, recall = utils.evaluate_model(model, MAX_CONTEXT_LENGTH, val_df1, x_val)
    print(f"F1: {f1}\t Precision: {precision}\t Recall: {recall}\t")

## LOAD tokenizer
#with open('tokenizer.json') as f:
#    data = json.load(f)
#    tokenizer = tokenizer_from_json(data)

##LOAD model
#reconstructed_model = keras.models.load_model("my_model")













