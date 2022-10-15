import os
import argparse
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import json
from scipy import sparse

m = Mystem()
nltk.download('stopwords')
tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()

def reduce_punct(text):
    sentences = ' '.join([sentence.translate(str.maketrans('', '', string.punctuation)) 
                          for sentence in text.replace('\n', ' ').split(' ')])
    sentences = re.sub('[0-9a-z]', '', sentences)
    return sentences

def lemmatize_text(text):
    lemmas = m.lemmatize(text)
    text = ''.join(lemmas)
    return text


def tokenize_text(text):
    text_tokens = word_tokenize(text)  
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentences = ' '.join(tokens_without_sw)
    return filtered_sentences


def preprocess_text(texts):
    for text in texts:
        text = text.lower()
        text = reduce_punct(text)
        text = lemmatize_text(text)
        text = tokenize_text(text)
    return texts

def create_corpus(data_dir):
    texts = []
    replies = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    
    for text in corpus:
        answers = json.loads(text)['answers']
        
        for answer in answers:
            author_values = np.array(answer['author_rating']['value'])
            if author_values:
                suitable_data = answers[np.argmax(author_values)]['text']
                texts.append(preprocess_text(suitable_data))
                replies.append(suitable_data)
            
    return texts, replies


def index_bm25(texts, k = 2, b = 0.75):
    count = count_vectorizer.fit_transform(texts)
    tf = count
    ti = tfidf_vectorizer.fit_transform(texts)
    idf = tfidf_vectorizer.idf_
    l_d = tf.sum(axis=1)
    avgdl = l_d.mean()

    rows, columns, values = [], [], []

    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        columns.append(j)
        values.append(idf[j] * (tf[i, j] * (k + 1)) / 
                      (tf[i, j] + k * (1 - b + b * l_d[i, 0] / avgdl)))

    matrix = sparse.csr_matrix((values, (rows, cols)))

    return matrix

def get_metrics(query, matrix):
    return np.dot(matrix, query.T)

def convert_query(query):
    query_vectorized = count_vectorizer.transform([preprocess_text(query)])
    return sparse.csr_matrix(query_vectorized)
            
def main(data_dir):
    texts, replies = create_corpus(data_dir)
    indexes = index_bm25(texts)
    print("Enter your query: ")
    users_query = input()
    
    if len(users_query) == 0:
        print('Sorry, your query is empty. Try again')
    else:
        query_vec = convert_query(users_query)
        similarity = np.argsort(get_metrics(users_query, indexes), axis=0)[::-1]
        search_results = np.array(replies)[similarity.ravel()]
        print(f'The most similar to your query can be found in these documents:')
        for result in search_results[:3]:
            print(f'{result[0]} : {result[1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                    default='',
                    required=True,
                    help='path to the data')
    args = parser.parse_args() 
    main(args.path)
