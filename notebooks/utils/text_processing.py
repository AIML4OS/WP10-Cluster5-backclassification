# Functions for pre-processing text for Random forest modelling
import string


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


def preprocess(df, text_variable, stopwords, nace_21_variable = "nace_21_code"):
    """
    Preprocesses text data by lowercasing, removing stopwords and punctuation, generating k-grams,
    and appending SN2025 classification.

    This function applies text cleaning steps,
    generates k-grams, and appends the 'sn2025_1' classification to the resulting string.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'navn', 'tekst', and 'sn2025_1' columns.

    Returns:
        pandas.Series: A Series of preprocessed text strings.

    Notes:
        - Assumes the existence of `remove_stopwords`, `remove_punctuation`, and `k_grams` functions.
    """
    tekst_pro = df[text_variable].str.lower()
    tekst_pro = tekst_pro.apply(remove_stopwords, stoppord_set = stopwords)
    tekst_pro = tekst_pro.apply(remove_punctuation)
    tekst_pro = tekst_pro.apply(k_grams)
    
    tekst_pro = tekst_pro + " " + df[nace_21_variable]
    
    return tekst_pro

def remove_stopwords(text, stoppord_set):
    return ' '.join([word for word in text.split() if word.lower() not in stoppord_set])

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

