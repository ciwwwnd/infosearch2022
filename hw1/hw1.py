import os
import argparse
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

m = Mystem()
nltk.download('stopwords')
vectorizer = CountVectorizer(analyzer='word')

def load_data(data_dir):
    texts = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding="utf8", errors='ignore') as f:
                text = f.read()
            texts.append(text)
    return texts


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
    X = vectorizer.fit_transform(preprocessed_texts)
    return X, vectorizer.get_feature_names()

def get_stats_info(indexed_results):
    matrix_freq = np.asarray(indexed_results[0].sum(axis=0)).ravel()
    features = np.array(indexed_results[1])
    
    max_ind = np.argmax(matrix_freq)
    print(f'Most frequent word is "{features[max_ind].upper()}", mentioned {matrix_freq[max_ind]} times.')

    min_ind = np.argmin(matrix_freq)
    print(f'Least frequent word is "{features[min_ind].upper()}", mentioned {matrix_freq[min_ind]} times.')

    print(f'''Words that are presented in all documents: {', '.join(features[np.all(indexed_results[0].toarray(), axis=0)])}''')
    
    
def main(data_dir):
    texts = load_data(data_dir)
    texts = preprocess_text(texts)
    indexes = index(texts)
    freqs_stats = get_stats_info(indexes)
    
    return freqs_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to the data directory')
    args = parser.parse_args()
    main(args.dir)

