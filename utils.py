from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix
import gensim.downloader as gloader
import numpy as np
import pandas as pd
import tensorflow as tf
import preprocess
from preprocess import nlp
from collections import OrderedDict
import chars2vec
import re



def get_word_listing(sentences):
    terms = [term for sentence in sentences for term in sentence.split()]
    return list(set(terms))


def tokenize(word_listing):
    '''
    Fit tokenizer, create word2idx e idx2word
    '''
    tokenizer = Tokenizer(filters = list())
    tokenizer.fit_on_texts(word_listing)
    indices = tokenizer.texts_to_sequences(word_listing)
    indices = [item for sublist in indices for item in sublist]
    word_to_idx = dict(zip(word_listing, indices))
    idx_to_word = dict(zip(indices, word_listing))

    return tokenizer, word_to_idx, idx_to_word


def get_co_occurrence_matrix(all_text, word_listing, word_to_idx, window_size=4): #jo messo 4, stava 1!
    '''
    Compute the co-occurrence matrix
    '''
    sentences = all_text
    rows, cols, data = [], [], []
    for sentence in sentences:
        words = sentence.split()
        for index, word in enumerate(words):
            row = word_to_idx.get(word)
            lower_limit = max(0, index-window_size)
            upper_limit = min(len(words), index+window_size+1)
            for i in range(lower_limit, upper_limit):
                if i != index:
                    col = word_to_idx.get(words[i])
                    rows.append(row)
                    cols.append(col)
                    data.append(1)

    return csr_matrix((data, (rows, cols)))


def compute_oov_embeddings(terms, word_to_idx, idx_to_word, co_occurrence_matrix, embedding_dim, embedding_model, random_strategy=False):
    '''
    Compute embedding for OOV terms.
    By default, neighboor strategy is used.
    '''
    embeddings = dict()
    vocabulary = embedding_model.key_to_index.keys()

    for term in terms:
        if random_strategy:
            embeddings[term] = np.random.rand(embedding_dim)
        else:
            count = 0
            s = np.zeros(embedding_dim)
            i = word_to_idx.get(term)
            co_occ_vec = co_occurrence_matrix.getrow(i)
            cols, values = co_occ_vec.indices, co_occ_vec.data
            for j in range(len(cols)):
                col, value = cols[j], values[j]
                if i != col:
                    neighbor = idx_to_word.get(col)
                    if neighbor in vocabulary:
                        count += value
                        s += embedding_model.get_vector(neighbor) * value
            if count == 0:
                embeddings[term] = np.random.rand(embedding_dim)
            else:
                embeddings[term] = s / count
    return embeddings




def get_embedding_matrix(dataframe, embedding_dim):
    '''
    Create the embedding matrix

    :param vocabulary: the vocabulary containing the words
    :param embedding_dim: the dimension of the embedding space
    :tokenizer_x: the tokenizer used to tokenize the data (x)

    :return
     emb_matrix: the embedding_matrix
    '''
    all_text = pd.concat([dataframe['context'], dataframe['question']], axis=0).unique()
    df_word_listing = get_word_listing(all_text)
    df_tokenizer, df_word_to_idx, df_idx_to_word = tokenize(df_word_listing)
    df_co_occurrence_matrix = get_co_occurrence_matrix(all_text, df_word_listing, df_word_to_idx)

    print("Loading GloVe embedding model...")
    embedding_model = gloader.load("glove-wiki-gigaword-{}".format(embedding_dim))

    embedding_dic = {key: embedding_model.get_vector(key)
                     for key in set(df_word_listing).intersection(embedding_model.key_to_index.keys())}
    print(f"There are {len(embedding_dic)} words for which we already know the embedding")
    oov = set(df_word_listing) - embedding_dic.keys()
    print(f"There are {len(oov)} oov words")

    print("Computing out-of-vocabulary embeddings...")
    embeddings_oov = compute_oov_embeddings(oov, df_word_to_idx, df_idx_to_word, df_co_occurrence_matrix,
                                            embedding_dim, embedding_model)
    embedding_dic = {**embedding_dic, **embeddings_oov}

    print("Computing embedding matrix...")
    emb_matrix = np.zeros((len(embedding_dic) + 2, embedding_dim)) #one for padding PAD and one for unknown UNK
    for k, v in embedding_dic.items():
        idx = df_word_to_idx.get(k)
        emb_matrix[idx] = v
    return emb_matrix, df_word_listing, df_tokenizer, df_word_to_idx, df_idx_to_word


def get_max_length(dataframe):
    len_context_tokens = [len(sentence.split()) for sentence in dataframe.context.unique()]
    MAX_CONTEXT_LENGTH = np.max(len_context_tokens)
    print(f'Max length for context is {MAX_CONTEXT_LENGTH}')
    print(f'Max length adopted for context is {int(MAX_CONTEXT_LENGTH * 1.1)}')

    len_text_tokens = [len(sentence.split()) for sentence in dataframe.text.values]
    MAX_TEXT_LENGTH = np.max(len_text_tokens)
    print(f'Max length for answer is {MAX_TEXT_LENGTH}')
    print(f'Max length adopted for answer is {int(MAX_TEXT_LENGTH * 1.1)}')

    len_question_tokens = [len(sentence.split()) for sentence in dataframe.question.values]
    MAX_QUESTION_LENGTH = np.max(len_question_tokens)
    print(f'Max length for question is {MAX_QUESTION_LENGTH}')
    print(f'Max length adopted for question is {int(MAX_QUESTION_LENGTH * 1.1)}')

    return int(MAX_CONTEXT_LENGTH * 1.1), int(MAX_TEXT_LENGTH * 1.1), int(MAX_QUESTION_LENGTH * 1.1)


def pad(df_values, tokenizer, max_length):
    x = [t.split() for t in df_values]
    x_encoded = tokenizer.texts_to_sequences(x)
    x_padded = pad_sequences(x_encoded, maxlen=max_length, padding='post')
    return x_padded



##TF_IDF
def compute_tf(df, MAX_CONTEXT_LENGTH):
    print("Computing normalized TF...")
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = df.context.values
    vectorizer = CountVectorizer(token_pattern=r"\S+")
    tf_context = vectorizer.fit_transform(corpus)

    tmp = pd.DataFrame(df.context.unique(), columns=['context'])
    tfs = []
    for i, row in tmp.context.iteritems():
        tokens = row.split()
        tfs.append([tf_context[i, vectorizer.vocabulary_[token]] for token in tokens])
    dict_context_tf = dict(zip(tmp.context, tfs))
    df_tf = df.context.apply(lambda x: dict_context_tf.get(x))
    df_tf_padded = pad_sequences(df_tf, maxlen=MAX_CONTEXT_LENGTH, padding='post', truncating='post')
    return df_tf_padded



#Exact match


def exact_match(df, MAX_CONTEXT_LENGTH):
    match = []
    for i in range(0, df.shape[0]):
        match1 = np.in1d(df.context[i].split(), df.question[i].split()).astype(int).reshape(1, -1)
        padded_match = pad_sequences(match1, padding="post", value=0, maxlen=MAX_CONTEXT_LENGTH, truncating='post')
        match.append(padded_match)
    return np.array(match)

def apply_exact_match(df, pipeline, MAX_CONTEXT_LENGTH):
    df2 = df.copy()
    df2, _ = preprocess.apply_preprocessing(df2, pipeline)
    # remove stopwords from question before computing exact match
    df2['question'] = df2['question'].apply(lambda x: preprocess.remove_stopwords(x))
    return exact_match(df2, MAX_CONTEXT_LENGTH).squeeze()

def compute_exact_match(df, MAX_CONTEXT_LENGTH):
    '''Original match'''
    print("Computing original exact match...")
    PREPROCESSING_PIPELINE1 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.strip_text]

    original_match = apply_exact_match(df, PREPROCESSING_PIPELINE1, MAX_CONTEXT_LENGTH).squeeze()

    '''Lowercase exact match'''
    print("Computing lowercase exact match...")
    PREPROCESSING_PIPELINE2 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.lower,
                               preprocess.strip_text]

    lowercase_match = apply_exact_match(df, PREPROCESSING_PIPELINE2, MAX_CONTEXT_LENGTH).squeeze()

    '''Lemmatized exact match'''
    print("Computing lemmatized exact match...")
    PREPROCESSING_PIPELINE3 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.lemmatization,
                               preprocess.lower,
                               preprocess.strip_text]

    lemmatized_match = apply_exact_match(df, PREPROCESSING_PIPELINE3, MAX_CONTEXT_LENGTH).squeeze()
    exact_match_input = np.stack((original_match, lowercase_match, lemmatized_match), axis=-1).astype(np.float32)
    return exact_match_input

#POS tags
def create_pos_dicts(pos_listing):
    print("Creating dictionaries for POS tags...")
    tag2idx = OrderedDict({tag: idx for idx, tag in enumerate(pos_listing)})
    idx2tag = OrderedDict({idx: tag for tag, idx in tag2idx.items()})
    # inserting pad token with idx=0 and moving first one to last
    tag2idx.update({(list(tag2idx.keys()))[0]: len(tag2idx)})
    idx2tag.update({len(idx2tag): (list(idx2tag.values()))[0]})
    tag2idx.move_to_end((list(tag2idx.keys()))[0], last=True)
    idx2tag.move_to_end(0, last=True)
    tag2idx.update({'<PAD>': 0})
    idx2tag.update({0: '<PAD>'})
    tag2idx.move_to_end('<PAD>', last=False)
    idx2tag.move_to_end(0, last=False)
    pos_listing = list(pos_listing)
    pos_listing.append(pos_listing.pop(0))
    pos_listing.insert(0, '<PAD>')
    return tag2idx, idx2tag


def compute_pos(df, tag2idx, MAX_CONTEXT_LENGTH):
    print("Computing POS tags...")
    docs = nlp.pipe(df.context, disable=["tok2vec", "ner", "lemmatizer"])
    postags = [[token.tag_ for token in doc] for doc in docs]
    #convert to integers using dict
    all = [[tag2idx[tag] for tag in context] for context in postags]
    print("Padding POS sequences...")
    padded_pos = pad_sequences(all, padding="post", value=tag2idx['<PAD>'],
                               maxlen=MAX_CONTEXT_LENGTH, truncating='post')
    dict_pos = dict(zip(df.context, padded_pos))
    pos_tmp = df.context.apply(lambda x: dict_pos.get(x))
    pos = np.array([t for t in pos_tmp])
    return pos


#NER tags
def create_ner_dicts(ner_listing):
    print("Creating dictionaries for NER tags...")
    ner2idx = OrderedDict({tag: idx for idx, tag in enumerate(ner_listing)})
    idx2ner = OrderedDict({idx: tag for tag, idx in ner2idx.items()})
    # inserting pad token with idx=0 and moving first one to last
    ner2idx.update({(list(ner2idx.keys()))[0]: len(ner2idx)})
    idx2ner.update({len(idx2ner): (list(idx2ner.values()))[0]})
    ner2idx.move_to_end((list(ner2idx.keys()))[0], last=True)
    idx2ner.move_to_end(0, last=True)
    ner2idx.update({'<PAD>': 0})
    idx2ner.update({0: '<PAD>'})
    ner2idx.move_to_end('<PAD>', last=False)
    idx2ner.move_to_end(0, last=False)
    ner_listing = list(ner_listing)
    ner_listing.append(ner_listing.pop(0))
    ner_listing.insert(0, '<PAD>')
    # insert none token for words without NER tag
    ner_listing.append('NONE')
    ner2idx.update({'NONE': len(ner2idx)})
    idx2ner.update({len(idx2ner): 'NONE'})
    return ner2idx, idx2ner

def compute_ner(df, ner2idx, MAX_CONTEXT_LENGTH):
    print("Computing NER tags...")
    docs = nlp.pipe(df.context, disable=["tok2vec", "tagger", "lemmatizer"])
    nertags = [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]
    all = []
    # convert to integers using dict
    for i in range(0, len(nertags)):
        k = 0
        sentence = np.full(shape=(len(df.context[i].split())), fill_value=ner2idx['NONE'])
        for first, second in nertags[i]:
            for word in first.split():
                for j in range(k, len(df.context[i].split())):
                    k = j
                    if word == df.context[i].split()[k]:
                        sentence[k] = ner2idx[second]
                        break
        all.append(sentence)
    all = np.array(all, dtype=object)
    print("Padding NER sequences...")
    padded_ner = pad_sequences(all, padding="post", value=ner2idx['<PAD>'],
                               maxlen=MAX_CONTEXT_LENGTH, truncating='post')

    dict_pos = dict(zip(df.context, padded_ner))
    ner_tmp = df.context.apply(lambda x: dict_pos.get(x))
    ner = np.array([t for t in ner_tmp])
    return ner


def get_char_embeddings(word_listing, word_to_idx):
    print("Computing character-level embeddings...")
    c2v_model = chars2vec.load_model('eng_50')
    char_embs = c2v_model.vectorize_words(word_listing)
    char_emb_dict = dict(zip(word_listing, char_embs))
    char_embedding_matrix = np.zeros((len(char_emb_dict) + 2, 50))  # +1 per il padding +2 per l'UNK
    for k, v in char_emb_dict.items():
        idx = word_to_idx.get(k)
        char_embedding_matrix[idx] = v
    return char_embedding_matrix


def compute_answers(predictions, df, df2):
    preds = np.argmax(predictions, -1)
    s_idx = preds[0]
    e_idx = preds[1]
    spans = []
    for i in range(preds.shape[1]):
        r2 = df2.loc[i].context.split()[s_idx[i]:e_idx[i]+1]
        s = '[^\w£$%]*?'.join(['[^\w\s£$%]*?'.join([re.escape(ch) for ch in word]) for word in r2])
        a = abs(len(df2.loc[i].context.split()) - len(df.loc[i].context.split()))
        idx = len(' '.join(df2.loc[i].context.split()[:s_idx[i]]))
        xre = re.search(s, df.context[i][max(0, idx-a):])
        spans.append(xre.group())
    return spans

def computing_predictions(model, train_df, val_df, test_df, x_tr, x_val, x_ts):
    print("Preprocessing on datasets...")
    print("Applying expand_contractions2, tokenization_spacy, remove_chars, split_alpha_num_sym and strip_text.")
    PREPROCESSING_PIPELINE_ = [preprocess.expand_contractions2,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.strip_text]
    tr_df2 = train_df.copy()
    tr_df2, tr_tmp1 = preprocess.apply_preprocessing(tr_df2, PREPROCESSING_PIPELINE_)
    val_df2 = val_df.copy()
    val_df2, val_tmp2 = preprocess.apply_preprocessing(val_df2, PREPROCESSING_PIPELINE_)
    ts_df2 = test_df.copy()
    ts_df2, ts_tmp2 = preprocess.apply_preprocessing(ts_df2, PREPROCESSING_PIPELINE_)

    print("Calculating predictions...")
    tr_predictions = model.predict(x_tr, batch_size=16)
    val_predictions = model.predict(x_val, batch_size=16)
    ts_predictions = model.predict(x_ts, batch_size=16)

    print("Computing answers...")
    tr_spans = compute_answers(tr_predictions, train_df, tr_df2)
    val_spans = compute_answers(val_predictions, val_df, val_df2)
    ts_spans = compute_answers(ts_predictions, test_df, ts_df2)
    tr_data = dict(zip(train_df.id, tr_spans))
    val_data = dict(zip(val_df.id, val_spans))
    ts_data = dict(zip(test_df.id, ts_spans))
    data = {**tr_data, **val_data, **ts_data}
    return data

def evaluate_model(model, MAX_CONTEXT_LENGTH, val_df1, x_val):
    print("Computing F1 score, precision and recall...")
    # create truth mask
    b = np.array(list(zip(val_df1.s_idx.values, val_df1.e_idx.values)))
    r = np.arange(MAX_CONTEXT_LENGTH)
    mask = (b[:, 0, None] <= r) & (b[:, 1, None] >= r)
    mask = mask.astype('int')
    s, e = model.predict(x_val, batch_size=16)

    # create predicted mask
    b2 = np.transpose([tf.argmax(s, -1), tf.argmax(e, -1)])
    mask2 = (b2[:, 0, None] <= r) & (b2[:, 1, None] >= r)
    mask2 = mask2.astype('int')

    product = tf.math.multiply(mask, mask2)
    product.shape
    shared = np.sum(product, axis=-1)
    predicted = np.sum(mask2, axis=-1)
    truth = np.sum(mask, axis=-1)

    precision = shared / predicted
    recall = shared / truth

    f1_sum = np.sum([2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0 for i in range(val_df1.shape[0])])

    return f1_sum/val_df1.shape[0], np.average(precision), np.average(recall)