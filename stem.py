#!/usr/bin/env python

import numpy as np
from stemming.porter2 import stem
import sys
from nltk.corpus import wordnet as wn
from collections import Counter
from sklearn.cluster import KMeans


# Prepare word by removing characters that are not alphanumeric
def prepare_word(word):
    return filter(str.isalnum, word.lower())

# Prepare document by removing non alphanumeric characters from words
def prepare_document(document):
    return map(prepare_word, document.split())

# Prepare documents (needs refactooor)
def prepare_documents(file, include_keywords):
    document_file = ""
    documents = []

    title = ""
    titles = []

    groups = []
    group = ""
    
    if include_keywords:
        group = "Group"
        title = "Title"

    for line in file:
        clean_line = line.strip()
        
        if not clean_line:
            temp_document = document_file.replace("-", " ")
            formated_document = prepare_document(temp_document)

            documents.append(formated_document)
            if not include_keywords:
                titles.append(title)
                groups.append(group)
                title = ""
                group = ""

            document_file = ""
        elif not group:
            group =  clean_line
        else:
            document_file += " " + clean_line
            if not title:
                title = clean_line

    formated_document = prepare_document(document_file)
    documents.append(formated_document)
    
    if not include_keywords:
        titles.append(title)
        groups.append(group)

    return documents, groups, titles

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

# Return idf for one term
def calculate_term_idf(total):
    return lambda term: 0.0 if number_of_documents_with_term(term) == 0 else np.log10(total / number_of_documents_with_term(term)) 

# Calculate idf
def calculate_idf(docs):
    number_of_documents = float(len(docs) - 1)
    return map(calculate_term_idf(number_of_documents), docs.T)

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
    #Check number of arguments
    if len(sys.argv) < 4:
        print "Error: Not enough arguments!!!"
        return

    # Read query from standard input
    mode = sys.argv[3]
    query = sys.argv[4:]
    
    grouping = mode == "group"

    # Return if query is nil
    if not grouping and len(query) == 0:
         print "Error: No query specified!!!"
         return

    if not grouping:
        # Gather propositions
        propositions = reduce(lambda x, y: x + similar_word(y), query, [])
        propositions = prepare_propositions(query, propositions)[:5]

        # Print interface
        query_string = reduce(lambda x, y: x + " " + y, query)
        print "<------ Query ------>"

        print query_string + "\n"

        print "<------ Similar queries ------>"
        for i, prop in enumerate(propositions):
            print str(i+1)+")", query_string + " " + prop[1]

        # Choose and extend query
        chosen_query = int(raw_input('\nChoose query (0 for original): '))
        if chosen_query > 0 and chosen_query <= len(propositions):
            query.append(propositions[chosen_query-1][1])
    
    # Read and prepare documents (query is treated like the last document)
    preformated = open(sys.argv[1], "r")
    
    formated, groups, titles = prepare_documents(preformated, False)
    
    if not grouping:
        formated += [query]

    stemmed = stem_documents(formated)

    # Read and prepare keywords
    key_preformated = open(sys.argv[2], "r")
    
    key_formated, y, x = prepare_documents(key_preformated, True)
    key_stemmed = np.unique(stem_documents(key_formated)[0])
    
    if not grouping:
        # Check if query is valid (contained in keywords)
        valid = len(filter(lambda q:stem(q) in key_stemmed, query)) > 0

        if not valid:
            print "Your query is invalid"
            return

    # Create similarity vector using tf-idf method
    bag = create_bag_of_words(stemmed, key_stemmed)
    tf = normalize_bag_of_words(bag)
    idf = calculate_idf(bag)
    tfidf = multiply(tf, idf)
    similar = sim(tfidf)

    if grouping:
        doc_similar = pre_compute_distances(tfidf)

        #Launch kMeans
        iterations = int(query[0]) if len(query) > 0 else 100
    
        if iterations > 0:
            kmeans = KMeans(n_clusters=9, random_state=0, max_iter=iterations).fit(doc_similar)

        dict = {}
        for i, gr in enumerate(groups):
            dict.setdefault(kmeans.labels_[i], [])
            dict[kmeans.labels_[i]] += [gr]

            for entry in dict:
                print (entry)
                for i, y in enumerate(dict[entry]):
                    print (y)
    else:

        # Prepare results and sort them by descending similarity
        sim_with_titles = map(lambda (i, sim): (groups[i], sim, titles[i]), enumerate(similar))
        result = filter(lambda (group, sim, title): sim >= 0, sim_with_titles)

        res_type = [('group', 'S100'), ('sim', float), ('title', 'S100')]

        result = np.asarray(result, dtype=res_type)
        result = np.sort(result, order='sim')

        print result[::-1]

# Find similar words for given word
def similar_word(word):
    if len(wn.synsets(word)) <= 0:
        return []
    
    ref = wn.synsets(word)[0]
    result = Counter()
    
    synsets = wn.synsets(word)
    for syn in synsets:
        for lem in syn.lemmas():
            value = lem.synset().path_similarity(ref)
            result[lem.name()] = max(value, result[lem.name()])

    result[word] = .0
    result = sorted([(result[k], k) for k in result], reverse=True)
    return result[:5]


# Ranks word based on path similarity
def rank_query_word(proposition):
    ref = wn.synsets(proposition)[0]
    def rank_word(word):
        return wn.synsets(word)[0].path_similarity(ref)
    return rank_word

def rank_proposition(query, proposition):
    ranks = map(rank_query_word(proposition), query)
    return reduce(lambda x,y: x+y, ranks)

# Prepares propositions
def prepare_proposition(query):
    def prepare(proposition):
        return (rank_proposition(query, proposition[1]), proposition[1].replace("_", " "))
    return prepare

def prepare_propositions(query, propositions):
    return sorted(map(prepare_proposition(query), propositions), reverse=True)

# Prepare matrix of distances
def pre_compute_distances_with_doc(tfidf):
    def compute(doc):
        return map(lambda doc2: 1 - cosinus_sim(doc)(doc2), tfidf)
    return compute

def pre_compute_distances(tfidf):
    return map(pre_compute_distances_with_doc(tfidf), tfidf)

if __name__ == "__main__":
    main()

