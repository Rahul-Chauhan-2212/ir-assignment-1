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
    create_inverted_index,
    create_biword_index,
    create_positional_index,
    create_dictionary
)
from modules.phrase_search import (
    biword_phrase_search,
    positional_phrase_search
)
from modules.preprocessing import preprocess
from modules.trees import (
    BinarySearchTree,
    BTree,
    build_balanced_bst
)

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
        "None",
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

            content = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

            documents = [
                line.strip()
                for line in content.split("\n")
                if line.strip()
            ]

        elif uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            text_column = st.selectbox(
                "Select Document Column",
                df.columns
            )

            documents = (
                df[text_column]
                .astype(str)
                .tolist()
            )

    except Exception as e:

        st.error(
            f"File Error: {e}"
        )

        st.stop()

# Initialise List of stop words, Stemmer and Lemmatizer Instances
stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

# Tabs for Document Previews, Preprocessing Results, Inverted Index
if documents:

    index = create_inverted_index(documents)

    biword_index = create_biword_index(documents)

    positional_index = create_positional_index(documents)

    # Dictionary Terms

    dictionary_terms = create_dictionary(index)

    # BST Creation

    bst = BinarySearchTree()

    sorted_terms = sorted(dictionary_terms)

    build_balanced_bst(bst, sorted_terms)

    # BTree Creation

    btree = BTree()

    for term in dictionary_terms:
        btree.insert(term)

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
            _, steps = preprocess(documents[selected_doc_pre], use_lowercase, remove_punctuation, remove_stopwords,
                                  hyphen_handling, tokenization, normalization_method)

            for step, value in steps.items():
                st.subheader(step)

                st.write(value)

    with tab3:
        # Shows Inverted Index for the document
        st.metric("Unique Terms", len(index))

        sample_index = dict(list(index.items())[:20])

        with st.expander("Sample Inverted Index"):
            st.json(sample_index)

        # Search the term from inverted index and provides document id
        search_term = st.text_input("Inspect a Term")

        if search_term:
            token, _ = preprocess(search_term, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling,
                                  tokenization, normalization_method)

            if token:
                term = token[0]

                if term in index:
                    st.success(
                        f"Found in {index[term]['df']} document(s)"
                    )

                    st.write(
                        "Document IDs:",
                        [x + 1 for x in index[term]["postings"]]
                    )
                else:
                    st.warning("Term not found in index")

    # Search Functionality
    st.header("Search Query")

    query = st.text_input(
        "Enter Search Query",
        key="search_query"
    )

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
                for token in query_tokens:
                    if token in index:
                        results.update(index[token]["postings"])

            elif retrieval_method == "Boolean Retrieval":
                posting_lists = []
                for token in query_tokens:
                    if token in index:
                        posting_lists.append(set(index[token]["postings"]))

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

    if st.button("Run Comparison"):

        stem_docs = []
        lemma_docs = []

        for doc in documents:
            doc = doc.lower()

            tokens = word_tokenize(doc)

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

    # Phase Query Processing
    st.header("Phrase Query Processing")

    phrase_query = st.text_input(
        "Enter Phrase Query",
        key="phrase_query"
    )

    if phrase_query:

        query_tokens, _ = preprocess(phrase_query, use_lowercase, remove_punctuation, remove_stopwords, hyphen_handling,
                                     tokenization, normalization_method)

        biword_results = biword_phrase_search(query_tokens, biword_index)

        positional_results = positional_phrase_search(query_tokens, positional_index)

        col1, col2 = st.columns(2)

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

    st.header(
        "BST vs B-Tree Comparison"
    )

    st.write(
        f"Dictionary Size : {len(dictionary_terms)} terms"
    )

    with st.expander(
            "Dictionary Sample"
    ):
        st.json(dictionary_terms[:20])

    sample_queries = st.text_input(
        "Enter Queries (comma separated)",
        key="tree_queries",
        value="batman,matrix,hero"
    )

    if st.button(
            "Compare Trees"
    ):
        queries = [

            q.strip()

            for q in sample_queries.split(",")

            if q.strip()
        ]

        comparison_df = compare_search_performance(
            queries,
            bst,
            btree,
            index
        )

        st.subheader(
            "Experimental Results"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        bst_avg = comparison_df[
            "BST Search (ms)"
        ].mean()

        bst_retrieval_avg = comparison_df[
            "BST Retrieval (ms)"
        ].mean()

        btree_avg = comparison_df[
            "BTree Search (ms)"
        ].mean()

        btree_retrieval_avg = comparison_df[
            "BTree Retrieval (ms)"
        ].mean()

        st.metric(
            "Average BST Search Time",
            round(
                bst_avg,
                6
            )
        )

        st.metric(
            "Average BST Retrival Time",
            round(
                bst_retrieval_avg,
                6
            )
        )

        st.metric(
            "Average BTree Search Time",
            round(
                btree_avg,
                6
            )
        )

        st.metric(
            "Average BTree Retrieval Time",
            round(
                btree_retrieval_avg,
                6
            )
        )

        st.subheader(
            "Inference"
        )

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
