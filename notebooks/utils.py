from typing import List, Tuple
import pandas as pd
import json
import spacy

from textacy.preprocessing import normalize_hyphenated_words, normalize_unicode, normalize_whitespace
from textacy.preprocessing import remove_accents, remove_punctuation
from textacy.spacier.doc_extensions import to_bag_of_words, to_bag_of_terms
from textacy.extract import semistructured_statements

from nltk.corpus import stopwords
from wordcloud import WordCloud

import matplotlib.pyplot as plt

text_processing_pipeline = [
    normalize_hyphenated_words, # remoção de hífens dentro de palavras
    normalize_unicode,          # tratamento de caractéres UNICODE
    normalize_whitespace,       # normalização de espaços em branco
    remove_accents              # remoção de acentos
]

pd.DataFrame

def apply_text_processing_pipeline(df: pd.Series) -> pd.Series:
    
    for func in text_processing_pipeline:
        df = df.apply(func)
    # Removendo string comum
    df = df.apply(lambda text: text.replace("[This review was collected as part of a promotion.]", ""))
    return df

def get_word_frequency_df(raw_text: spacy.language.Doc) -> pd.DataFrame:

    word_dict = to_bag_of_words(raw_text, filter_stops=True, filter_punct=True, as_strings=True)

    word_freq_df = pd.DataFrame(index=word_dict.keys(), data=word_dict.values(), columns=["freq"])
    word_freq_df.drop("-PRON-", axis=0, inplace=True) # special token
    word_freq_df.sort_values(by="freq", ascending=False, inplace=True)

    return word_freq_df

def get_ngram_df(raw_text: spacy.language.Doc, n: tuple = (2,3,4)) -> pd.DataFrame:

    ngrams_dict = to_bag_of_terms(raw_text, ngrams=n, filter_stops=True, filter_punct=True, as_strings=True)
    
    ngram_df = pd.DataFrame(index=ngrams_dict.keys(), data=ngrams_dict.values(), columns=["freq"])
    ngram_df.drop(index="review be collect", inplace=True) # frase recorrente
    ngram_df.sort_values(by="freq", ascending=False, inplace=True)

    return ngram_df

def build_wordcloud(text: str, additional_stop_words: List[str], color_map: str = "binary", output_file: str = "wordcloud.png", save_file: bool = False) -> None:

    stop_words = stopwords.words("english")
    stop_words.extend([word.lower() for word in additional_stop_words])

    wc = WordCloud(colormap=color_map, max_words=200, stopwords=stop_words, background_color='white', width=1600, height=800)
    cloud = wc.generate(text.lower())

    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    # plt.close()

    if save_file: wc.to_file(output_file)

def get_statements(parsed_doc: spacy.language.Doc, possible_subjects: List[str]) -> List[tuple]:

    statements = []

    for subject in possible_subjects:
        statements.extend(list(semistructured_statements(parsed_doc, subject)))

    return statements

def serialize_statements(statements: Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span], output_path: str) -> None:

    serializable_object = []

    for statement in statements:
        serializable_object.append(
            [
                statement[0].text,
                statement[1].text,
                statement[2].text
            ]
        )
    
    with open(output_path, "w") as f:
        json.dump(serializable_object, f, indent=2)
    
