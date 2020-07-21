import pandas as pd
from textacy.preprocessing import normalize_hyphenated_words, normalize_unicode, normalize_whitespace
from textacy.preprocessing import remove_accents, remove_punctuation

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

    return df