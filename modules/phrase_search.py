# ==========================================
# Phrase Search using Biword Indexes
# ==========================================
def biword_phrase_search(query_tokens, biword_index):
    """
    Returns documents based on query from biword index
    :param query_tokens: User Query Tokens
    :param biword_index:  Biword Index
    :return: Documents
    """

    if len(query_tokens) < 2:
        return []

    query_biwords = []

    for i in range(len(query_tokens) - 1):
        query_biwords.append(f"{query_tokens[i]} {query_tokens[i + 1]}")

    result_sets = []

    for biword in query_biwords:

        if biword not in biword_index:
            return []

        result_sets.append(set(biword_index[biword]))

    if not result_sets:
        return []

    return list(set.intersection(*result_sets))


# ==========================================
# Phrase Search using Positional Indexes
# ==========================================
def positional_phrase_search(query_tokens, positional_index):
    """
    Returns documents based on query from positional index
    :param query_tokens: User Query Token
    :param positional_index:  Positional Index
    :return: Documents
    """

    if len(query_tokens) == 0:
        return []

    if query_tokens[0] not in positional_index:
        return []

    candidate_docs = set(positional_index[query_tokens[0]].keys())

    for token in query_tokens[1:]:
        if token not in positional_index:
            return []
        candidate_docs &= set(positional_index[token].keys())

    final_results = []

    for doc_id in candidate_docs:

        first_positions = positional_index[query_tokens[0]][doc_id]

        found = False

        for pos in first_positions:

            match = True

            for offset in range(1, len(query_tokens)):

                current_token = (query_tokens[offset])

                if (pos + offset) not in positional_index[current_token][doc_id]:
                    match = False
                    break

            if match:
                found = True
                break

        if found:
            final_results.append(doc_id)

    return final_results
