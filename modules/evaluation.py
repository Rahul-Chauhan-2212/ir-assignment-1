import time
import pandas as pd


# ==========================================
# Performance Comparison
# ==========================================

def compare_search_performance(
        queries,
        bst,
        btree,
        index
):
    results = []

    for query in queries:
        start = time.perf_counter()

        bst.search(
            bst.root,
            query
        )

        bst_search_time = (
                time.perf_counter()
                - start
        )

        start = time.perf_counter()

        postings = []

        if query in index:
            postings = index[query]["postings"]

        for doc_id in postings:
            _ = doc_id

        bst_retrieval_time = (
                time.perf_counter()
                - start
        )

        start = time.perf_counter()

        btree.search(query)

        btree_search_time = (
                time.perf_counter()
                - start
        )

        start = time.perf_counter()

        postings = []

        if query in index:
            postings = index[query]["postings"]

        for doc_id in postings:
            _ = doc_id

        btree_retrieval_time = (
                time.perf_counter()
                - start
        )

        results.append({

            "Query": query,

            "BST Search (ms)":
                bst_search_time * 1000,

            "BST Retrieval (ms)":
                bst_retrieval_time * 1000,

            "BTree Search (ms)":
                btree_search_time * 1000,

            "BTree Retrieval (ms)":
                btree_retrieval_time * 1000
        })

    return pd.DataFrame(results)
