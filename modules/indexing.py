from collections import defaultdict

import streamlit as st

from modules.preprocessing import preprocess_documents
from modules.tolerant_retrieval import create_kgram_index


# ==========================================
# Inverted Index Creation
# ==========================================
def create_inverted_index(processed_docs):
    """
    Creates Inverted Index from docs
    :param processed_docs: Processed Documents
    :return: Inverted Index
    """
    inverted_index = {}
    for doc in processed_docs:
        doc_id = doc["doc_id"]
        tokens = doc["tokens"]
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
def create_biword_index(processed_docs):
    """
    Creates Biword Index from docs
    Example:
    "dark knight rises"
    dark knight -> docID
    knight rises -> docID
    :param processed_docs: Processed Documents
    :return: Biword Index
    """

    biword_index = defaultdict(list)

    for doc in processed_docs:
        doc_id = doc["doc_id"]
        tokens = doc["tokens"]

        for i in range(len(tokens) - 1):

            biword = f"{tokens[i]} {tokens[i + 1]}"

            if doc_id not in biword_index[biword]:
                biword_index[biword].append(doc_id)

    return dict(biword_index)


def create_positional_index(processed_docs):
    """
    Creates Positional Index from docs
    term ->docID ->positions
    :param processed_docs: Processed Documents
    :return: Positional Index
    """

    positional_index = defaultdict(lambda: defaultdict(list))

    for doc in processed_docs:
        doc_id = doc["doc_id"]
        tokens = doc["tokens"]

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


@st.cache_resource
def build_indexes(documents, use_lowercase=True,
                  remove_punctuation=True,
                  remove_stopwords=True,
                  hyphen_handling=True,
                  tokenization=True,
                  normalization_method="None"):
    """
    Builds all inverted, biword and positional indexes and dictionary from documents
    :param documents: Documents
    :param use_lowercase: If Lowercase enabled
    :param remove_punctuation: If Punctuation Removal enabled
    :param remove_stopwords: If Stopwords Removal enabled
    :param hyphen_handling: If Hyphen Removal enabled
    :param tokenization: If Tokenization enabled
    :param normalization_method: If Normalization Method -> Stemming or Lemmatization
    :return: All three indexes
    """

    processed_documents = preprocess_documents(documents, use_lowercase, remove_punctuation, remove_stopwords,
                                               hyphen_handling, tokenization, normalization_method)
    inverted_index = create_inverted_index(processed_documents)
    dictionary_terms = create_dictionary(inverted_index)
    return (
        inverted_index,
        create_biword_index(processed_documents),
        create_positional_index(processed_documents),
        dictionary_terms,
        create_kgram_index(dictionary_terms)
    )
