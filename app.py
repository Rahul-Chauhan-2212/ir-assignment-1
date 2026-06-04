import re
from collections import defaultdict

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK Library Downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="",
    layout="wide"
)

st.title("Information Retrieval System")

# Preprocessing Options
st.sidebar.header("Preprocessing Options")

use_lowercase = st.sidebar.checkbox("Lowercase Text", value=True)

remove_punctuation = st.sidebar.checkbox("Remove Punctuation", value=True)

remove_stopwords = st.sidebar.checkbox("Remove Stopwords", value=True)

hyphen_handling = st.sidebar.checkbox("Hyphen Handling", value=True)

tokenization = st.sidebar.checkbox("Tokenization", value=True)

use_stemming = st.sidebar.checkbox("Apply Stemming")

use_lemmatization = st.sidebar.checkbox("Apply Lemmatization")

# Retrieval Options
st.sidebar.header("Retrieval Options")

retrieval_method = st.sidebar.selectbox(
    "Select Retrieval Method",
    [
        "Boolean Retrieval",
        "Keyword Matching"
    ]
)

# Document Upload containing documents
uploaded_file = st.file_uploader("Upload TXT or CSV Dataset", type=["txt", "csv"])

documents = []

if uploaded_file:

    st.success("Dataset Uploaded Successfully")

    if uploaded_file.name.endswith(".txt"):

        content = uploaded_file.read().decode("utf-8")

        documents = [
            line.strip()
            for line in content.split("\n")
            if line.strip()
        ]

    elif uploaded_file.name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        documents = (
            df.iloc[:, 0]
            .astype(str)
            .tolist()
        )

# Initialise List of stop words, Stemmer and Lemmatizer Instances
stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()


# Preprocessing
def preprocess(text):
    """
    Preprocess the text based on the selected Preprocessing methods
    Returns Final Tokens and Intermediate Steps Results
    :param text: Document
    :return: final_tokens, intermediate_steps
    """
    intermediate_steps = {}

    intermediate_steps["Original"] = text

    # Lower case the input document
    if use_lowercase:
        text = text.lower()

    intermediate_steps["After Lowercase"] = text

    # Remove Hyphens
    if hyphen_handling:
        text = text.replace("-", " ")

    intermediate_steps["After Hyphen Handling"] = text

    # Remove Punctuations
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    intermediate_steps["After Punctuation Removal"] = text

    # Tokenize the document
    tokens = text.split()

    intermediate_steps["After Tokenization"] = tokens

    # Remove Stop words from tokens
    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

    intermediate_steps["After Stopword Removal"] = tokens

    # Perform Stemming
    stemmed_tokens = [
        stemmer.stem(token)
        for token in tokens
    ]

    intermediate_steps["After Stemming"] = stemmed_tokens

    # Perform Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    intermediate_steps["After Lemmatization"] = lemmatized_tokens

    final_tokens = tokens

    if use_stemming:
        final_tokens = stemmed_tokens

    elif use_lemmatization:
        final_tokens = lemmatized_tokens

    return final_tokens, intermediate_steps


# Inverted Index Creation
def create_inverted_index(docs):
    """
    Create Inverted Index using documents
    :param docs: Documents
    :return: inverted_index
    """
    inverted_index = defaultdict(list)

    for doc_id, doc in enumerate(docs):

        tokens, _ = preprocess(doc)

        for token in tokens:

            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)

    return dict(inverted_index)


# Tabs for Document Previews, Preprocessing Results, Inverted Index
if documents:

    index = create_inverted_index(documents)

    st.header("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Documents", len(documents))

    with col2:
        st.metric("Vocabulary Size", len(index))

    tab1, tab2, tab3 = st.tabs(
        [
            "Document Preview",
            "Preprocessing",
            "Inverted Index"
        ]
    )

    with tab1:
        # Shows the selected document content
        selected_doc = st.selectbox("Select Document", range(len(documents)), format_func=lambda x: f"Document {x + 1}")
        st.text_area("Document Content", documents[selected_doc], height=200)

    with tab2:
        # Shows Intermediate Outputs for Preprocessing Options
        selected_doc_pre = st.selectbox("Select Document for Analysis", range(len(documents)),
                                        format_func=lambda x: f"Document {x + 1}", key="pre_doc")
        with st.expander("Show Intermediate Outputs"):
            _, steps = preprocess(documents[selected_doc_pre])

            for step, value in steps.items():
                st.subheader(step)

                st.write(value)

    with tab3:
        # Shows Inverted Index for the document
        st.metric("Unique Terms", len(index))

        st.subheader("Sample Inverted Index")

        sample_index = dict(list(index.items())[:20])

        st.json(sample_index)

        # Search the term from inverted index and provides document id
        search_term = st.text_input("Inspect a Term")

        if search_term:
            token, _ = preprocess(search_term)

            if token:
                term = token[0]

                if term in index:
                    st.success(f"Found in {len(index[term])} document(s)")
                    st.write("Document IDs:", [x + 1 for x in index[term]])
                else:
                    st.warning("Term not found in index")

# Search Functionality
st.header("Search Query")

query = st.text_input("Enter Search Query")

if st.button("Search"):
    if not documents:
        st.warning("Please upload a dataset first.")
    elif not query:
        st.warning("Please enter a search query.")
    else:
        # Inverted index for documents
        index = create_inverted_index(documents)
        # Apply same preprocess on the queries
        query_tokens, query_steps = (preprocess(query))

        # To show preprocssing happened on Seacrh Query
        with st.expander("Query Processing Details"):
            st.json(query_steps)

        results = set()

        # Retrieval Based on Selected Option
        if retrieval_method == "Keyword Matching":
            for token in query_tokens:
                if token in index:
                    results.update(index[token])

        elif retrieval_method == "Boolean Retrieval":
            posting_lists = []
            for token in query_tokens:
                if token in index:
                    posting_lists.append(set(index[token]))

            if posting_lists:
                results = set.intersection(*posting_lists)

        st.header("Search Results")

        if not results:
            st.error("No matching documents found.")
        else:
            st.success(f"{len(results)} document(s) found")

            # Show Search Results doc id and document
            for doc_id in sorted(results):
                with st.expander(f"Document {doc_id + 1}"):
                    st.write(documents[doc_id])

# Stemming vs Lemmatization Comparison
st.header("Stemming vs Lemmatization Comparison")

if documents:

    if st.button("Run Comparison"):

        stem_docs = []
        lemma_docs = []

        for doc in documents:
            doc = doc.lower()

            tokens = doc.split()

            tokens = [
                t
                for t in tokens
                if t not in stop_words
            ]

            # Create Stemmed Docs from created stemmed tokens
            stem_docs.append(
                " ".join(
                    [
                        stemmer.stem(t)
                        for t in tokens
                    ]
                )
            )

            # Create Lemmatized docs from created lemmatized tokens
            lemma_docs.append(
                " ".join(
                    [
                        lemmatizer.lemmatize(t)
                        for t in tokens
                    ]
                )
            )

        # Using TF-IDF Vectorization Technique to create vector metrix
        vectorizer = TfidfVectorizer()

        stem_matrix = vectorizer.fit_transform(stem_docs)

        lemma_matrix = vectorizer.fit_transform(lemma_docs)

        # Calculate Cosine Similarity Mean for both Stemming and Lemmatization
        stem_score = (cosine_similarity(stem_matrix).mean())

        lemma_score = (cosine_similarity(lemma_matrix).mean())

        st.info("TF-IDF Vectorization used and Cosine Similarity Score calculated for Stemming and Lemmatization")

        # Create Comparison table and Conclusion
        comparison_df = pd.DataFrame({
            "Technique":
                [
                    "Stemming",
                    "Lemmatization"
                ],
            "Similarity Score":
                [
                    round(stem_score, 4),
                    round(lemma_score, 4)
                ]
        })

        st.table(comparison_df)

        if lemma_score > stem_score:
            st.success(
                """
                Conclusion:
                Lemmatization performs better on this dataset because it preserves semantic meaning
                while reducing words to their dictionary form.
                """
            )
        else:
            st.success(
                """
                Conclusion:
                Stemming performs better on this dataset because it reduces vocabulary size more aggressively.
                """
            )
