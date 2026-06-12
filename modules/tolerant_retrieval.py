import re
from collections import defaultdict


# =====================================================
# K-GRAM INDEX
# =====================================================

def create_kgram_index(dictionary_terms, k=2):
    """
    Create K-Gram index
    """

    kgram_index = defaultdict(set)

    for term in dictionary_terms:

        padded = f"${term}$"

        for i in range(len(padded) - k + 1):
            gram = padded[i:i + k]

            kgram_index[gram].add(term)

    return kgram_index


# =====================================================
# WILDCARD RETRIEVAL
# =====================================================

def wildcard_search(pattern, dictionary_terms):
    """
    Convert wildcard query to regex.
    ```
    Example:
    bat*
    *man
    b*tman
    """

    regex_pattern = "^" + pattern.replace("*", ".*") + "$"

    regex = re.compile(
        regex_pattern
    )

    matches = []

    for term in dictionary_terms:

        if regex.match(term):
            matches.append(term)

    return sorted(matches)


# =====================================================
# EDIT DISTANCE
# =====================================================

def edit_distance(word1, word2):
    """
    Levenshtein Distance
    """
    m = len(word1)

    n = len(word2)

    dp = [
        [0] * (n + 1)
        for _ in range(m + 1)
    ]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            cost = 0

            if word1[i - 1] != word2[j - 1]:
                cost = 1

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]


# =====================================================
# SPELLING CORRECTION
# =====================================================

def spelling_correction(query, dictionary_terms):
    """
    Return closest dictionary term.
    """
    best_word = None

    best_distance = float("inf")

    for term in dictionary_terms:

        distance = edit_distance(query, term)

        if distance < best_distance:
            best_distance = distance

            best_word = term

    return best_word, best_distance


# =====================================================
# SOUNDEX
# =====================================================

def soundex(word):
    """
    Soundex encoding
    """
    word = word.upper()

    if not word:
        return ""

    mappings = {

        "BFPV": "1",

        "CGJKQSXZ": "2",

        "DT": "3",

        "L": "4",

        "MN": "5",

        "R": "6"
    }

    encoded = word[0]

    previous = ""

    for char in word[1:]:

        digit = ""

        for letters, code in mappings.items():

            if char in letters:
                digit = code
                break

        if digit != previous:
            encoded += digit

        previous = digit

    encoded = encoded.replace("", "")

    encoded = (
            encoded +
            "000"
    )[:4]

    return encoded


# =====================================================
# PHONETIC SEARCH
# =====================================================

def phonetic_search(query, dictionary_terms):
    """
    Find words with same Soundex.
    """

    query_code = soundex(query)

    matches = []

    for term in dictionary_terms:

        if soundex(term) == query_code:
            matches.append(term)

    return matches


def apply_tolerant_retrieval(query_tokens, dictionary_terms, index):
    """
    Apply tolerant retrieval techniques before keyword matching.

    Techniques:
    - Wildcard
    Expansion
    - Spelling
    Correction
    - Edit
    Distance
    - Phonetic
    Correction
    """

    corrected_tokens = []

    correction_messages = []

    for token in query_tokens:

        # --------------------------------
        # Wildcard Query
        # --------------------------------

        if "*" in token:
            matches = wildcard_search(
                token,
                dictionary_terms
            )

            corrected_tokens.extend(matches)

            correction_messages.append(
                f"Wildcard '{token}' → {matches}"
            )

            continue

        # --------------------------------
        # Exact Match
        # --------------------------------

        if token in index:
            corrected_tokens.append(token)

            continue

        # --------------------------------
        # Edit Distance / Spelling
        # --------------------------------

        suggestion, distance = (
            spelling_correction(
                token,
                dictionary_terms
            )
        )

        if suggestion and distance <= 2:
            corrected_tokens.append(
                suggestion
            )

            correction_messages.append(
                f"Corrected '{token}' → '{suggestion}'"
            )

            continue

        # --------------------------------
        # Phonetic Correction
        # --------------------------------

        phonetic_matches = (
            phonetic_search(
                token,
                dictionary_terms
            )
        )

        if phonetic_matches:
            corrected_tokens.append(
                phonetic_matches[0]
            )

            correction_messages.append(
                f"Phonetic Match '{token}' → '{phonetic_matches[0]}'"
            )

            continue

        corrected_tokens.append(token)

    return (
        corrected_tokens,
        correction_messages
    )
