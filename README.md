# Information Retrieval System

The application demonstrates the complete IR pipeline including text preprocessing, indexing, retrieval models, phrase
querying, tolerant retrieval, and search data structure comparison.

### Install Dependencies

```
pip install -r requirements.txt
```

### Run Application

```
streamlit run app.py
```

# Features

## Dataset Upload and Search Interface

* Upload `.txt` and `.csv` datasets.
* View uploaded documents.
* Enter search queries through the Streamlit interface.
* Configure preprocessing and retrieval options.
* Display intermediate and final outputs.

---

## Text Preprocessing

The system supports the following preprocessing operations:

### Lowercasing

Converts all text to lowercase.

Example:

```text
BATMAN → batman
```

### Tokenization

Splits text into individual tokens.

Example:

```text
Batman fights Joker
→
["batman", "fights", "joker"]
```

### Stop Word Removal

Removes common English stop words.

Example:

```text
the batman and the joker
→
["batman", "joker"]
```

### Hyphen Handling

Converts hyphenated terms into separate words.

Example:

```text
state-of-the-art
→
state of the art
```

### Stemming

Uses the Porter Stemmer to reduce words to root forms.

Example:

```text
running → run
wolves → wolv
```

### Lemmatization

Uses WordNet Lemmatizer to obtain dictionary forms.

Example:

```text
cars → car
children → child
```

### Stemming vs Lemmatization Comparison

The system compares retrieval effectiveness using:

* Stemming
* Lemmatization

Comparison is performed using query-based TF-IDF cosine similarity scores.

---

# Task C: Index Construction and Query Processing

## Inverted Index

Stores:

```text
Term
↓
Document Frequency
↓
Posting List
```

Example:

```text
batman
DF = 3
Postings = [1, 4, 7]
```

---

## Biword Index

Supports phrase retrieval using adjacent term pairs.

Example:

```text
dark knight
space mission
```

---

## Positional Index

Stores term positions within documents for phrase queries.

Example:

```text
batman:
Document 5 → [3, 12, 28]
```

---

## Boolean Retrieval

Current Implementation only performs AND Matching

```text
batman joker -> batman AND joker
```

---

# Search Structure Comparison

The system compares:

* Balanced Binary Search Tree (BST)
* B-Tree Simulation

Metrics:

* Search Time
* Retrieval Time

Experimental results are displayed directly within the application.

---

# Tolerant Retrieval

The search engine supports imperfect user queries using multiple tolerant retrieval techniques.

## Wildcard Retrieval

Examples:

```text
bat*
*man
b*tman
```

---

## Spelling Correction

Examples:

```text
batmn → batman
wizrad → wizard
matix → matrix
```

---

## Edit Distance Correction

Uses Levenshtein Edit Distance to identify the closest matching term from the dictionary.

---

## K-Gram Index

Used for candidate generation during spelling correction and tolerant retrieval.

---

## Phonetic Retrieval (Soundex)

Retrieves terms that sound similar.

Examples:

```text
matriks → matrix
batmaan → batman
```

---

# Project Structure

```text
ir-assignment-1/
│
├── app.py
│
├── modules/
│   ├── preprocessing.py
│   ├── indexing.py
│   ├── phrase_search.py
│   ├── tolerant_retrieval.py
│   ├── trees.py
│   ├── evaluation.py
│
├── dataset/
│
├── requirements.txt
│
└── README.md
```

# Test Queries

## Keyword Search

```text
batman
joker
wizard
matrix
hero
```

---

## Boolean Retrieval

```text
batman joker
```

---

## Phrase Queries

```text
dark knight
space mission
crime family
```

---

## Wildcard Queries

```text
bat*
*man
b*tman
```

---

## Spelling Correction

```text
batmn
wizrad
matix
jokerr
```

---

## Phonetic Retrieval

```text
matriks
batmaan
jonker
```

---

# Technologies Used

* Python
* Streamlit
* NLTK
* NumPy
* Pandas
* Scikit-Learn

---
