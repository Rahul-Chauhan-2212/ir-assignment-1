import re
from collections import defaultdict

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="",
    layout="wide"
)

st.title("Information Retrieval System")

st.sidebar.header("Preprocessing Options")

use_lowercase = st.sidebar.checkbox("Lowercase Text", value=True)

remove_punctuation = st.sidebar.checkbox("Remove Punctuation", value=True)

tokenization = st.sidebar.checkbox("Tokenization", value=True)

st.sidebar.header("Retrieval Options")

retrieval_method = st.sidebar.selectbox(
    "Select Retrieval Method",
    [
        "Boolean Retrieval",
        "Keyword Matching"
    ]
)

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


def preprocess(text):
    intermediate_steps = {}

    intermediate_steps["Original"] = text

    if use_lowercase:
        text = text.lower()

    intermediate_steps["After Lowercase"] = text

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    intermediate_steps["After Punctuation Removal"] = text

    if tokenization:
        tokens = text.split()
    else:
        tokens = [text]

    intermediate_steps["Tokens"] = tokens

    return tokens, intermediate_steps


def create_inverted_index(docs):
    inverted_index = defaultdict(list)

    for doc_id, doc in enumerate(docs):

        tokens, _ = preprocess(doc)

        for token in tokens:

            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)

    return dict(inverted_index)


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
        selected_doc = st.selectbox("Select Document", range(len(documents)), format_func=lambda x: f"Document {x + 1}")
        st.text_area("Document Content", documents[selected_doc], height=200)

    with tab2:
        selected_doc_pre = st.selectbox("Select Document for Analysis", range(len(documents)),
                                        format_func=lambda x: f"Document {x + 1}", key="pre_doc")
        with st.expander("Show Intermediate Outputs"):
            _, steps = preprocess(documents[selected_doc_pre])

            for step, value in steps.items():
                st.subheader(step)

                st.write(value)

    with tab3:
        st.metric("Unique Terms", len(index))
        search_term = st.text_input("Inspect a Term")

        if search_term:
            token, _ = preprocess(search_term)
            token = token[0]

            if token in index:
                st.success(f"Found in {len(index[token])} document(s)")
                st.write("Document IDs:", [x + 1 for x in index[token]])
            else:
                st.warning("Term not found in index")

st.header("Search Query")

query = st.text_input("Enter Search Query")

if st.button("Search"):
    if not documents:
        st.warning("Please upload a dataset first.")
    elif not query:
        st.warning("Please enter a search query.")
    else:
        index = create_inverted_index(documents)

        query_tokens, query_steps = (preprocess(query))

        with st.expander("Query Processing Details"):
            st.json(query_steps)

        results = set()

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

            for doc_id in sorted(results):
                with st.expander(f"Document {doc_id + 1}"):
                    st.write(documents[doc_id])
