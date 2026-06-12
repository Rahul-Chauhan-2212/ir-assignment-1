from collections import defaultdict

from modules.preprocessing import preprocess


# ==========================================
# Inverted Index Creation
# ==========================================
def create_inverted_index(docs, use_lowercase=True,
                          remove_punctuation=True,
                          remove_stopwords=True,
                          hyphen_handling=True,
                          tokenization=True,
                          normalization_method="None"):
    """
    Creates Inverted Index from docs
    :param docs: Documents
    :param use_lowercase: If Lowercase enabled
    :param remove_punctuation: If Punctuation Removal enabled
    :param remove_stopwords: If Stopwords Removal enabled
    :param hyphen_handling: If Hyphen Removal enabled
    :param tokenization: If Tokenization enabled
    :param normalization_method: Normalization Method -> Stemming or Lemmatization
    :return: Inverted Index
    """
    inverted_index = {}
    for doc_id, doc in enumerate(docs):

        tokens, _ = preprocess(doc, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling, tokenization,
                               normalization_method)

        for token in tokens:

            if token not in inverted_index:
                inverted_index[token] = {
                    "df": 0,
                    "postings": []
                }

            if doc_id not in inverted_index[token]["postings"]:
                inverted_index[token]["postings"].append(doc_id)

                inverted_index[token]["df"] += 1

    return inverted_index


# ==========================================
# Biword Index Creation
# ==========================================
def create_biword_index(docs, use_lowercase=True,
                        remove_punctuation=True,
                        remove_stopwords=True,
                        hyphen_handling=True,
                        tokenization=True,
                        normalization_method="None"):
    """
    Creates Biword Index from docs
    Example:
    "dark knight rises"
    dark knight -> docID
    knight rises -> docID
    :param docs: Documents
    :param use_lowercase: If Lowercase enabled
    :param remove_punctuation: If Punctuation Removal enabled
    :param remove_stopwords: If Stopwords Removal enabled
    :param hyphen_handling: If Hyphen Removal enabled
    :param tokenization: If Tokenization enabled
    :param normalization_method: If Normalization Method -> Stemming or Lemmatization
    :return: Biword Index
    """

    biword_index = defaultdict(list)

    for doc_id, doc in enumerate(docs):

        tokens, _ = preprocess(doc, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling, tokenization,
                               normalization_method)

        for i in range(len(tokens) - 1):

            biword = f"{tokens[i]} {tokens[i + 1]}"

            if doc_id not in biword_index[biword]:
                biword_index[biword].append(doc_id)

    return dict(biword_index)


def create_positional_index(docs, use_lowercase=True,
                            remove_punctuation=True,
                            remove_stopwords=True,
                            hyphen_handling=True,
                            tokenization=True,
                            normalization_method="None"):
    """
    Creates Positional Index from docs
    term ->docID ->positions
    :param docs: Documents
    :param use_lowercase: If Lowercase enabled
    :param remove_punctuation: If Punctuation Removal enabled
    :param remove_stopwords: If Stopwords Removal enabled
    :param hyphen_handling: If Hyphen Removal enabled
    :param tokenization: If Tokenization enabled
    :param normalization_method: Normalization Method -> Stemming or Lemmatization
    :return: Positional Index
    """

    positional_index = defaultdict(lambda: defaultdict(list))

    for doc_id, doc in enumerate(docs):
        tokens, _ = preprocess(doc, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling, tokenization,
                               normalization_method)

        for position, token in enumerate(tokens):
            positional_index[token][doc_id].append(position)

    return positional_index


def create_dictionary(index):
    """
    Extract
    all
    unique
    terms
    from inverted index.
    """

    return list(index.keys())
