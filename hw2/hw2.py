import os
import argparse
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.spatial.distance import cosine

m = Mystem()
nltk.download('stopwords')
vectorizer = TfidfVectorizer()

def load_data(data_dir):
    texts = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding="utf8", errors='ignore') as f:
                text = f.read()
            texts.append(text)
    return texts

def get_filenames(data_dir):
    file_names = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            file_names.append(name)
    return file_names

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

def index(preprocessed_texts):
    X = vectorizer.fit_transform(preprocessed_texts).toarray()
    return X

def get_metrics(vec1, vec2):
    return (1 - cosine(vec1, vec2))

def convert_query(query, index_matrix):
    query_vectorized = vectorizer.transform([preprocess_text(query)]).toarray()
    cosine_applied = np.apply_along_axis(lambda x: get_metrics(x, query_vectorized[0]), 1, index_matrix)
    return cosine_applied
      
def get_results_tfidf(query, f_names, indexes):
    results = list(zip(f_names, convert_query(query, indexes)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main(data_dir):
    texts = load_data(data_dir)
    f_names = get_filenames(data_dir)
    texts = preprocess_text(texts)
    indexes = index(texts)
    print("Enter your query: ")
    users_query = input()
    if len(users_query) == 0:
        print('Sorry, your query is empty. Try again')
    else:
        search_results = get_results_tfidf(users_query, f_names, indexes)
        print(f'The most similar to your query can be found in these documents:')
        for result in search_results[:3]:
            print(f'{result[0]} : {result[1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to the data directory')
    args = parser.parse_args() 
    main(args.dir)