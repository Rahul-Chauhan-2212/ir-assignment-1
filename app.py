import time

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.evaluation import (
    compare_search_performance
)
from modules.indexing import (
    build_indexes
)
from modules.phrase_search import (
    biword_phrase_search,
    positional_phrase_search
)
from modules.preprocessing import preprocess
from modules.tolerant_retrieval import (
    apply_tolerant_retrieval
)
from modules.trees import (
    BinarySearchTree,
    BTree,
    build_balanced_bst
)

# ========================================================================================================
# NOTE: The code in broken module(individual Py files) for each step or process to reduce code complexity)
#        The app.py contains usage of those steps and shows the Streamlit Results
# =========================================================================================================


# NLTK Library Downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")
nltk.download("punkt_tab")

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

normalization_method = st.sidebar.radio(
    "Normalization Method",
    [
        "Stemming",
        "Lemmatization"
    ]
)

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
index = {}
biword_index = {}
positional_index = {}

if uploaded_file:

    st.success("Dataset Uploaded Successfully")

    try:

        if uploaded_file.name.endswith(".txt"):

            content = uploaded_file.read().decode("utf-8", errors="ignore")

            documents = [
                line.strip()
                for line in content.split("\n")
                if line.strip()
            ]

        elif uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            text_column = st.selectbox("Select Document Column", df.columns)

            documents = (
                df[text_column]
                .astype(str)
                .tolist()
            )

    except Exception as e:
        st.error(f"File Error: {e}")
        st.stop()

# Initialize List of stop words, Stemmer and Lemmatizer Instances
stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

# Tabs for Document Previews, Preprocessing Results, Inverted Index
if documents:

    start = time.perf_counter()
    print("Start Indexes Creation")
    # Index creation process is present in indexing.py
    (
        inverted_index,
        biword_index,
        positional_index,
        dictionary_terms,
        kgram_index
    ) = build_indexes(documents, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling, tokenization,
                      normalization_method)
    print("End Index Creation")
    end = time.perf_counter()
    print("Total Time: {} seconds".format(end - start))

    # BST and BTree Parts are present in trees.py
    # BST Creation
    bst = BinarySearchTree()

    sorted_terms = sorted(dictionary_terms)

    build_balanced_bst(bst, sorted_terms)

    # BTree Creation
    btree = BTree()

    for term in dictionary_terms:
        btree.insert(term)

    # Tabs and Columns to show Uploaded Documents, Preprocessing Results and Inverted Index
    st.header("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Documents", len(documents))

    with col2:
        st.metric("Vocabulary Size", len(inverted_index))

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
            _, steps = preprocess(documents[selected_doc_pre], use_lowercase, remove_punctuation, remove_stopwords,
                                  hyphen_handling, tokenization, normalization_method)

            for step, value in steps.items():
                st.subheader(step)

                st.write(value)

    with tab3:
        # Shows Inverted Index for the document
        st.metric("Unique Terms", len(inverted_index))

        sample_index = dict(list(inverted_index.items())[:20])

        with st.expander("Sample Inverted Index"):
            st.json(sample_index)

        # Search the term from inverted index and provides document id
        search_term = st.text_input("Inspect a Term")

        if search_term:
            token, _ = preprocess(search_term, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling,
                                  tokenization, normalization_method)

            if token:
                term = token[0]

                if term in inverted_index:
                    st.success(f"Found in {inverted_index[term]['df']} document(s)")

                    st.write(
                        "Document IDs:",
                        [x + 1 for x in inverted_index[term]["postings"]]
                    )
                else:
                    st.warning("Term not found in index")

    # ==========================================
    #         Search Functionality
    # ==========================================
    st.header("Search Query")

    query = st.text_input("Enter Search Query", key="search_query")

    if st.button("Search"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            # Apply same preprocess on the queries
            query_tokens, query_steps = preprocess(query, use_lowercase, remove_punctuation, remove_stopwords,
                                                   hyphen_handling, tokenization, normalization_method)

            # To show preprocssing happened on Seacrh Query
            with st.expander("Query Processing Details"):
                st.json(query_steps)

            results = set()

            # Retrieval Based on Selected Option
            if retrieval_method == "Keyword Matching":
                # Tolerant Retrieval Methods can be applied for Keyword Searching as we are not searching for exact match here like Boolean Retrieval
                query_tokens, correction_messages = (
                    apply_tolerant_retrieval(
                        query_tokens,
                        dictionary_terms,
                        kgram_index
                    )
                )

                if correction_messages:
                    st.info(
                        "\n".join(
                            correction_messages
                        )
                    )
                for token in query_tokens:
                    if token in inverted_index:
                        results.update(inverted_index[token]["postings"])

            elif retrieval_method == "Boolean Retrieval":
                posting_lists = []
                for token in query_tokens:
                    if token in inverted_index:
                        posting_lists.append(set(inverted_index[token]["postings"]))

                # This currently checks implements only AND(Exact Match) i.e. terms1 term2 term3 -> term1 AND term2 AND term3
                # This can be modified to have OR and NOT. For simplicity not implemented boolean query search
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

    # ==========================================
    # Stemming vs Lemmatization Comparison
    # ==========================================
    st.header("Stemming vs Lemmatization Comparison")

    if st.button("Run Comparison"):

        if not query:
            st.warning("Please enter a search query first.")
        else:

            # ------------------------------------
            # Create Stemmed Documents
            # ------------------------------------

            stem_docs = []

            for doc in documents:
                tokens = word_tokenize(doc.lower())

                tokens = [
                    t
                    for t in tokens
                    if t not in stop_words
                ]

                stem_docs.append(
                    " ".join(
                        [
                            stemmer.stem(t)
                            for t in tokens
                        ]
                    )
                )

            # ------------------------------------
            # Create Lemmatized Documents
            # ------------------------------------

            lemma_docs = []

            for doc in documents:
                tokens = word_tokenize(doc.lower())

                tokens = [
                    t
                    for t in tokens
                    if t not in stop_words
                ]

                lemma_docs.append(
                    " ".join(
                        [
                            lemmatizer.lemmatize(t)
                            for t in tokens
                        ]
                    )
                )

            # ------------------------------------
            # Process Query for Stemming
            # ------------------------------------

            query_tokens = word_tokenize(query.lower())

            query_tokens = [
                t
                for t in query_tokens
                if t not in stop_words
            ]

            stem_query = " ".join(
                [
                    stemmer.stem(t)
                    for t in query_tokens
                ]
            )

            lemma_query = " ".join(
                [
                    lemmatizer.lemmatize(t)
                    for t in query_tokens
                ]
            )

            # ------------------------------------
            # TF-IDF Representation
            # ------------------------------------

            stem_vectorizer = TfidfVectorizer()

            stem_doc_matrix = stem_vectorizer.fit_transform(stem_docs)

            stem_query_vector = stem_vectorizer.transform([stem_query])

            stem_similarity = cosine_similarity(stem_query_vector, stem_doc_matrix)[0]

            # ------------------------------------

            lemma_vectorizer = TfidfVectorizer()

            lemma_doc_matrix = lemma_vectorizer.fit_transform(lemma_docs)

            lemma_query_vector = lemma_vectorizer.transform([lemma_query])

            lemma_similarity = cosine_similarity(lemma_query_vector, lemma_doc_matrix)[0]

            # ------------------------------------
            # Comparison Metrics
            # ------------------------------------

            stem_score = stem_similarity.max()

            lemma_score = lemma_similarity.max()

            comparison_df = pd.DataFrame({

                "Technique": [
                    "Stemming",
                    "Lemmatization"
                ],

                "Top Retrieval Similarity": [
                    round(stem_score, 4),
                    round(lemma_score, 4)
                ]
            })

            st.table(comparison_df)

            st.subheader("Query Used")

            st.code(query)

            st.subheader("Processed Queries")

            st.write(f"Stemmed Query: {stem_query}")

            st.write(f"Lemmatized Query: {lemma_query}")

            if lemma_score > stem_score:

                st.success(
                    """
                    Conclusion:

                    Lemmatization produced a higher
                    retrieval similarity score for
                    the selected query.

                    Therefore lemmatization is more
                    suitable for this dataset because
                    it preserves semantic meaning
                    while normalizing terms.
                    """
                )

            elif stem_score > lemma_score:

                st.success(
                    """
                    Conclusion:

                    Stemming produced a higher
                    retrieval similarity score for
                    the selected query.

                    Therefore stemming is more
                    suitable for this dataset because
                    it aggressively reduces term
                    variations and increases matches.
                    """
                )

            else:

                st.info(
                    """
                    Both techniques produced
                    similar retrieval effectiveness
                    for the selected query.
                    """
                )

    # ==========================================
    #            Phrase Query Processing
    # ==========================================
    st.header("Phrase Query Processing")

    phrase_query = st.text_input("Enter Phrase Query", key="phrase_query")

    if phrase_query:

        query_tokens, _ = preprocess(phrase_query, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling,
                                     tokenization, normalization_method)

        biword_results = biword_phrase_search(query_tokens, biword_index)

        positional_results = positional_phrase_search(query_tokens, positional_index)

        col1, col2 = st.columns(2)

        # Tabs to Shows Biword and Positional Index Results
        with col1:

            st.subheader("Biword Index Results")

            st.write("Matching Documents:", len(biword_results))

            for doc_id in biword_results:
                st.success(f"Document {doc_id + 1}")

                st.write(documents[doc_id])

        with col2:

            st.subheader("Positional Index Results")

            st.write("Matching Documents:", len(positional_results))

            for doc_id in positional_results:
                st.success(f"Document {doc_id + 1}")

                st.write(documents[doc_id])

        with st.expander("View Biword Index"):
            st.json(dict(list(biword_index.items())[:20]))

        with st.expander("View Positional Index"):
            sample = {}

            count = 0

            for term, posting in positional_index.items():

                sample[term] = dict(posting)

                count += 1

                if count == 10:
                    break

            st.json(sample)

        st.subheader("Inference")

        st.info(
            """
            Biword Index stores pairs of adjacent words.
        
            It is faster and smaller than a positional index.
        
            However, it may produce false positives
            for long phrase queries because only
            consecutive word pairs are checked.
        
            Positional Index stores exact positions
            of every term in every document.
        
            Therefore it verifies that all query
            terms occur in the exact sequence and
            position.
        
            Positional Index is more accurate for
            phrase searching, although it requires
            additional storage.
            """
        )

    # ==========================================
    # BST vs B-Tree Comparison
    # ==========================================

    st.header("BST vs B-Tree Comparison")

    st.write(f"Dictionary Size : {len(dictionary_terms)} terms")

    with st.expander("Dictionary Sample"):
        st.json(dictionary_terms[:20])

    sample_queries = st.text_input("Enter Queries (comma separated)", key="tree_queries", value="batman,matrix,hero")

    if st.button("Compare Trees"):
        queries = [

            q.strip()

            for q in sample_queries.split(",")

            if q.strip()
        ]

        comparison_df = compare_search_performance(queries, bst, btree, index)

        st.subheader("Experimental Results")

        st.dataframe(comparison_df, use_container_width=True)

        bst_avg = comparison_df["BST Search (ms)"].mean()

        bst_retrieval_avg = comparison_df["BST Retrieval (ms)"].mean()

        btree_avg = comparison_df["BTree Search (ms)"].mean()

        btree_retrieval_avg = comparison_df["BTree Retrieval (ms)"].mean()

        st.metric("Average BST Search Time", round(bst_avg, 6))

        st.metric("Average BST Retrival Time", round(bst_retrieval_avg, 6))

        st.metric("Average BTree Search Time", round(btree_avg, 6))

        st.metric("Average BTree Retrieval Time", round(btree_retrieval_avg, 6))

        st.subheader("Inference")

        st.info(
            """
            BST stores terms in a binary hierarchy.

            Search complexity is O(log n)
            for a balanced tree.

            B-Trees store multiple keys
            in a node and therefore require
            fewer comparisons.

            For large dictionaries B-Trees
            are generally faster and more
            efficient.

            B-Trees are widely used in
            databases and search engines
            because they minimize disk
            accesses.

            Therefore B-Trees are generally
            preferred for large-scale
            Information Retrieval systems.
            """
        )
