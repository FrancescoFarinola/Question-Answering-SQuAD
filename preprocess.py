import symspellpy
from symspellpy import SymSpell, Verbosity
import pkg_resources
sym_spell = SymSpell(max_dictionary_edit_distance=2)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, 0, 1)

#import unidecode
import pandas as pd

import nltk
nltk.download('wordnet')

import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['parser', 'senter', 'attribute_ruler'])

import re
from functools import reduce
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


WHITESPACES_RE = re.compile("\s+")
CHARS_TO_SPACE = re.compile("[–—\-\\/\[\]\(\)\+:]")
CHARS_TO_REMOVE = re.compile("[^\w\s£\$%]")


#def unicode_decode(text):
#    return unidecode.unidecode(text)


def expand_contractions(text):
    """
    Expands contracted words
    """
    text = re.sub(r"won't\b", "will not", text)
    text = re.sub(r"can't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'s\b", " s", text)
    text = re.sub(r"'d\b", " would", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'m\b", " am", text)

    # string operation
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')
    return text


def expand_contractions2(text):
    text = re.sub(r"won't\b", "wo n't", text)
    text = re.sub(r"can't\b", "ca n't", text)
    text = re.sub(r"n't\b", " n't", text)
    text = re.sub(r"'re\b", " 're", text)
    text = re.sub(r"\'s\b", " 's", text)
    text = re.sub(r"'d\b", " 'd", text)
    text = re.sub(r"'ll\b", " 'll", text)
    # text = re.sub(r"\'t", " not", text)
    text = re.sub(r"'ve\b", " 've", text)
    text = re.sub(r"'m\b", " 'm", text)

    # string operation
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')
    return text


def tokenization_spacy(text):
    # split text in tokens and then join it
    return ' '.join([token.text for token in nlp(text, disable=["tagger", "ner", "lemmatizer"])])


def remove_chars(text):
    text = CHARS_TO_SPACE.sub(' ', text)  # split words
    return CHARS_TO_REMOVE.sub('', text)  # do not split words


def split_alpha_num_sym(text):
    # split alphabetic from numeric characters and symbols and vice-versa
    text = re.sub(r'(\d)([a-zA-Z£\$%])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z£\$%])(\d)', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([£\$%])', r'\1 \2', text)
    text = re.sub(r'([£\$%])([a-zA-Z])', r'\1 \2', text)
    return text


def spell_correction(text):
    # max edit distance: 2
    results = [t if (t.isnumeric() or t.istitle() or len(t) < 5)
               else sym_spell.lookup(t, Verbosity.TOP, max_edit_distance=2,
                                     include_unknown=True)[0].term
               for t in text.split()]
    return ' '.join(results)


def lemmatization(text):
    # lemmatization
    return ' '.join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])


def lower(text):
    # lowercase
    return text.lower()


def strip_text(text):
    # strip text
    return WHITESPACES_RE.sub(' ', text)


def remove_stopwords(text, stopwords = nlp.Defaults.stop_words):
    return " ".join([w for w in text.split() if not(w in stopwords)])


def preprocessing(text, preprocessing_pipeline):
    return reduce(lambda text, f: f(text), preprocessing_pipeline, text)


def apply_preprocessing(df, pipeline, text=True):
    # get distinct contexts
    tmp = pd.DataFrame(df.context.unique(), columns=['context'])
    # apply preprocessing on distinct contexts
    tmp.context = tmp.context.apply(lambda x: preprocessing(x, pipeline))
    # mapping:  not_preprocessed_context -> preprocessed_context
    dict_context = dict(zip(df.context.unique(), tmp.context))
    # substitute not_preprocessed_context with preprocessed_context
    df.context = df.context.apply(lambda x: dict_context.get(x))

    if text:
        df['text'] = df['text'].apply(lambda x: preprocessing(x, pipeline))
    df['question'] = df['question'].apply(lambda x: preprocessing(x, pipeline))
    return df, tmp

"""
def clean_text(dataframe):
    PREPROCESSING_PIPELINE = [expand_contractions,
                              tokenization_spacy,
                              normalize_accents,
                              remove_punctuation,
                              filter_out_uncommon_symbols,
                              split_digit_alpha,
                              spell_correction,
                              lemmatization,
                              lower,
                              strip_text]
    df1 = dataframe.copy()
    # get distinct contexts
    tmp = pd.DataFrame(dataframe.context.unique(), columns=['context'])
    # apply preprocessing on distinct contexts

    tmp.context = tmp.context.apply(lambda x: preprocessing(x, PREPROCESSING_PIPELINE))
    # mapping:  not_preprocessed_context -> preprocessed_context
    dict_context = dict(zip(dataframe.context.unique(), tmp.context))
    # substitute not_preprocessed_context with preprocessed_context
    df1.context = df1.context.apply(lambda x: dict_context.get(x))

    df1['text'] = df1['text'].apply(lambda x: preprocessing(x, PREPROCESSING_PIPELINE))
    df1['question'] = df1['question'].apply(lambda x: preprocessing(x, PREPROCESSING_PIPELINE))
    return df1
"""

