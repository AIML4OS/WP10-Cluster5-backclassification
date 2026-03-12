# Pre-processing of text variables for ML model

import random
import string
import pandas as pd
import numpy as np


def k_grams(txt, k=3):
    """
    Generates k-grams (substrings of length k) from a given text, replacing spaces with underscores.

    This function slides a window of length `k` over the input string and extracts overlapping
    substrings. Spaces within each k-gram are replaced with underscores. The resulting k-grams
    are concatenated into a single string, separated by spaces.

    Args:
        txt (str): The input text string.
        k (int, optional): The length of each k-gram. Defaults to 3.

    Returns:
        str: A space-separated string of k-grams with spaces replaced by underscores.
    """
    string_tot = ""
    for i in range(len(txt) - k + 1):
        txt_cur = txt[i:(i+k)]
        txt_cur = txt_cur.replace(" ", "_")
        string_tot = string_tot + txt_cur + " "
    return string_tot

def preprocess(df, company_text, company_nace21, stopwords=None):
    """
    Preprocesses text data by lowercasing, removing stopwords and punctuation, generating k-grams,
    and appending SN2025 classification.

    This function combines the 'navn' and 'tekst' columns, applies text cleaning steps,
    generates k-grams, and appends the 'sn2025_1' classification to the resulting string.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'navn', 'tekst', and 'sn2025_1' columns.

    Returns:
        pandas.Series: A Series of preprocessed text strings.

    Notes:
        - Assumes the existence of `remove_stopwords`, `remove_punctuation`, and `k_grams` functions.
    """
    # Convert string to lower
    tekst_pro = df[company_text].str.lower()

    # Remove stopwords
    if not stopwords:
        stopwords = norwegian_stopwords
    tekst_pro = tekst_pro.apply(remove_stopwords(stopwords))

    # Remove punctuation
    tekst_pro = tekst_pro.apply(remove_punctuation)

    # Convert to character grams
    tekst_pro = tekst_pro.apply(k_grams)
    
    # Add in NACE 2.1
    tekst_pro = tekst_pro + " " + df[company_nace21]
    
    return tekst_pro


def remove_stopwords(text, stopwords):
    return ' '.join([word for word in text.split() if word.lower() not in stopwords])


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


norwegian_stopwords = set([
    "as",
    "å",
    "alle",
    "andre",
    "at",
    "av",
    "både",
    "båe",
    "bare",
    "begge",
    "ble",
    "blei",
    "bli",
    "blir",
    "blitt",
    "bort",
    "bra",
    "bruke",
    "da",
    "då",
    "de",
    "deg",
    "dei",
    "deim",
    "deira",
    "deires",
    "dem",
    "den",
    "denne",
    "der",
    "dere",
    "deres",
    "det",
    "dette",
    "di",
    "din",
    "disse",
    "dit",
    "ditt",
    "du",
    "dykk",
    "dykkar",
    "eg",
    "ein",
    "eit",
    "eitt",
    "eller",
    "elles",
    "en",
    "ene",
    "eneste",
    "enhver",
    "enn",
    "er",
    "et",
    "ett",
    "etter",
    "få",
    "for",
    "før",
    "fordi",
    "forsøke",
    "først",
    "fra",
    "fram",
    "gå",
    "gjorde",
    "gjøre",
    "god",
    "ha",
    "hadde",
    "han",
    "hans",
    "har",
    "hennar",
    "henne",
    "hennes",
    "her",
    "hit",
    "hjå",
    "ho",
    "hoe",
    "honom",
    "hoss",
    "hossen",
    "hun",
    "hva",
    "hvem",
    "hver",
    "hvilke",
    "hvilken",
    "hvis",
    "hvor",
    "hvordan",
    "hvorfor",
    "i",
    "ikke",
    "ikkje",
    "ingen",
    "ingi",
    "inkje",
    "inn",
    "innen",
    "inni",
    "ja",
    "jeg",
    "kan",
    "kom",
    "korleis",
    "korso",
    "kun",
    "kunne",
    "kva",
    "kvar",
    "kvarhelst",
    "kven",
    "kvi",
    "kvifor",
    "lage",
    "lang",
    "lik",
    "like",
    "må",
    "man",
    "mange",
    "måte",
    "me",
    "med",
    "medan",
    "meg",
    "meget",
    "mellom",
    "men",
    "mens",
    "mer",
    "mest",
    "mi",
    "min",
    "mine",
    "mitt",
    "mot",
    "mye",
    "mykje",
    "nå",
    "når",
    "ned",
    "nei",
    "no",
    "noe",
    "noen",
    "noka",
    "noko",
    "nokon",
    "nokor",
    "nokre",
    "ny",
    "og",
    "også",
    "om",
    "opp",
    "oss",
    "over",
    "på",
    "rett",
    "riktig",
    "så",
    "samme",
    "sånn",
    "seg",
    "selv",
    "si",
    "sia",
    "sidan",
    "siden",
    "sin",
    "sine",
    "sist",
    "sitt",
    "sjøl",
    "skal",
    "skulle",
    "slik",
    "slutt",
    "so",
    "som",
    "somme",
    "somt",
    "start",
    "stille",
    "tid",
    "til",
    "tilbake",
    "um",
    "under",
    "upp",
    "ut",
    "uten",
    "være",
    "vært",
    "var",
    "vår",
    "vart",
    "varte",
    "ved",
    "verdi",
    "vere",
    "verte",
    "vi",
    "vil",
    "ville",
    "vite",
    "vore",
    "vors",
    "vort",
])