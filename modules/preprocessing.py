import re
import time

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()


# ==========================================
#  Preprocess the documents
# ==========================================
def preprocess(text,
               use_lowercase=True,
               remove_punctuation=True,
               remove_stopwords=True,
               hyphen_handling=True,
               tokenization=True,
               normalization_method="None"
               ):
    """
    Apply Preprocessing Steps on documents and return Tokens and Intermediate Steps Results
    :param text: Text to be preprocessed
    :param use_lowercase: If Lowercase to be applied
    :param remove_punctuation: If Punctuation Removal to be applied
    :param remove_stopwords: If Stopwords to be Removed
    :param hyphen_handling: If Hyphen Handling to be applied
    :param tokenization: If Tokenization to be applied
    :param normalization_method: Normalization Method -> Lemmatization OR Stemming
    :return: Tokens and Intermediate Steps Results
    """

    intermediate_steps = {}

    intermediate_steps["Original"] = text

    if use_lowercase:
        text = text.lower()

    intermediate_steps["After Lowercase"] = text

    if hyphen_handling:
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)

    intermediate_steps["After Hyphen Handling"] = text

    if remove_punctuation:
        text = re.sub(r'\[\d+\]', '', text)  # remove citations
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

    intermediate_steps["After Punctuation Removal"] = text

    if tokenization:
        tokens = word_tokenize(text)
    else:
        tokens = [text]

    intermediate_steps["After Tokenization"] = tokens

    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

    intermediate_steps["After Stopword Removal"] = tokens

    final_tokens = tokens

    if normalization_method == "Stemming":

        final_tokens = [
            stemmer.stem(token)
            for token in tokens
        ]

        intermediate_steps["After Stemming"] = final_tokens

    elif normalization_method == "Lemmatization":

        final_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
        ]

        intermediate_steps["After Lemmatization"] = final_tokens

    return final_tokens, intermediate_steps


@st.cache_resource
def preprocess_documents(documents, use_lowercase=True,
                         remove_punctuation=True,
                         remove_stopwords=True,
                         hyphen_handling=True,
                         tokenization=True,
                         normalization_method="None"):
    """
    Preprocess Documents
    :param documents: Preprocessed Documents
    :param use_lowercase: If Lowercase to be applied
    :param remove_punctuation: If Punctuation Removal to be applied
    :param remove_stopwords: If Stopwords to be Removed
    :param hyphen_handling: If Hyphen Handling to be applied
    :param tokenization: If Tokenization to be applied
    :param normalization_method: If Normalization Method to be applied
    :return: Proceessed Documents
    """
    processed_docs = []
    start = time.perf_counter()
    print("Preprocessing Documents...")
    for doc_id, doc in enumerate(documents):
        tokens, _ = preprocess(doc, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling, tokenization,
                               normalization_method)

        processed_docs.append({"doc_id": doc_id,
                               "tokens": tokens})

    end = time.perf_counter()
    print("Preprocessing Documents Complete in {} seconds".format(end - start))

    return processed_docs
