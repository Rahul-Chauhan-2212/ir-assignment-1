import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()


def preprocess(
        text,
        use_lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        hyphen_handling=True,
        tokenization=True,
        normalization_method="None"
):
    """
    Complete preprocessing pipeline.
    """

    intermediate_steps = {}

    intermediate_steps["Original"] = text

    if use_lowercase:
        text = text.lower()

    intermediate_steps["After Lowercase"] = text

    if hyphen_handling:
        text = re.sub(
            r"(?<=\w)-(?=\w)",
            " ",
            text
        )

    intermediate_steps["After Hyphen Handling"] = text

    if remove_punctuation:
        text = re.sub(
            r"[^\w\s]",
            "",
            text
        )

    intermediate_steps[
        "After Punctuation Removal"
    ] = text

    if tokenization:
        tokens = word_tokenize(text)
    else:
        tokens = [text]

    intermediate_steps[
        "After Tokenization"
    ] = tokens

    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

    intermediate_steps[
        "After Stopword Removal"
    ] = tokens

    final_tokens = tokens

    if normalization_method == "Stemming":

        final_tokens = [
            stemmer.stem(token)
            for token in tokens
        ]

        intermediate_steps[
            "After Stemming"
        ] = final_tokens

    elif normalization_method == "Lemmatization":

        final_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
        ]

        intermediate_steps[
            "After Lemmatization"
        ] = final_tokens

    return final_tokens, intermediate_steps
