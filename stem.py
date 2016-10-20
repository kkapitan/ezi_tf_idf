#!/usr/bin/env python

import numpy as np
from stemming.porter2 import stem
import sys

# Prepare word by removing characters that are not alphanumeric
def prepare_word(word):
    return filter(str.isalnum, word.lower())

# Prepare document by removing non alphanumeric characters from words
def prepare_document(document):
    return map(prepare_word, document.split())

# Prepare documents (needs refactooor)
def prepare_documents(file):
    document_file = ""
    documents = []

    title = ""
    titles = []
    
    for line in file:
        clean_line = line.strip()
        
        if clean_line:
            document_file += " " + clean_line
            if not title:
                title = clean_line
        else:
            temp_document = document_file.replace("-", " ")
            formated_document = prepare_document(temp_document)

            documents.append(formated_document)
            titles.append(title)

            document_file = ""
            title = ""

    formated_document = prepare_document(document_file)
    documents.append(formated_document)
    titles.append(title)
    return documents, titles

# Stem document
def stem_document(document):
    return map(stem, document)

# Stem documents
def stem_documents(documents):
    return map(stem_document, documents)

# Create bag of word representation for document
def bag_of_words_for_document(key_words):
    return lambda document: map(document.count, key_words)

# Create bag of word representation for documents
def create_bag_of_words(documents, key_words):
    return np.asarray(map(bag_of_words_for_document(key_words), documents))

# Normalize document by dividing each word frequency by the maximal frequency
def normalize_document(document):
    max_value = max(document)
    return map(lambda value: float(value)/max_value, document)

# Normalize all documents
def normalize_bag_of_words(documents):
    return np.asarray(map(normalize_document, documents))

# Return number of document containing given term
def number_of_documents_with_term(term):
    return len(filter(lambda x: x > 0, term[:-1]))

# Calculate idf
def calculate_idf(docs):
    number_of_documents = float(len(docs) - 1)
    return map(lambda t: np.log10(number_of_documents / number_of_documents_with_term(t)), docs.T)

# Multiply tf by idf
def multiply(tf, idf):
    return map(lambda t: t * idf, tf)

# Calculate magnitude of vector
def magnitude(row):
    return np.sqrt(sum(row*row))

# Calculate similarities matrix
def sim(tfidf):
    return map(cosinus_sim(tfidf[-1]), tfidf[:-1])

# Calculate similarity for given vector
def cosinus_sim(q):
    m_q = magnitude(q)
    def sim(d):
        m_d = magnitude(d)
        return np.dot(d, q) / (m_q * m_d) if m_d * m_q else 0
    return sim

def main():
    # Read query from standard input
    query = sys.argv[1:]
    
    # Read and prepare documents (query is treated like the last document)
    preformated = open("documents.txt", "r")
    formated, titles = prepare_documents(preformated)
    formated += [query]
    stemmed = stem_documents(formated)

    # Read and prepare keywords
    key_preformated = open("keywords.txt", "r")
    key_formated, x = prepare_documents(key_preformated)
    key_stemmed = np.unique(stem_documents(key_formated)[0])
    
    # Check if query is valid (contained in keywords)
    valid = len(filter(lambda q:stem(q) in key_stemmed, query)) > 0

    if not valid:
        return
    
    # Create similarity vector using tf-idf method
    bag = create_bag_of_words(stemmed, key_stemmed)
    tf = normalize_bag_of_words(bag)
    idf = calculate_idf(bag)
    tfidf = multiply(tf, idf)
    similar = sim(tfidf)
    
    # Prepare results and sort them by descending similarity
    sim_with_titles = map(lambda (i, sim): (sim, titles[i]), enumerate(similar))
    result = filter(lambda (sim, title): sim > 0, sim_with_titles)

    res_type = [('sim', float), ('title', 'S100')]

    result = np.asarray(result, dtype=res_type)
    result = np.sort(result, order='sim')

    print result[::-1]

if __name__ == "__main__":
    main()
