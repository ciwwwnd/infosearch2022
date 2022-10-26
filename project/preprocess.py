import re
import nltk
from nltk.corpus import stopwords
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

nltk.download('stopwords')
stop_words = set(stopwords.words("russian"))

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = []
    for token in text.split():
        token = token.lower()
        if token.isalpha() and token not in stop_words:
            token = morph.normal_forms(token.strip())[0]
            tokens.append(token)
    return ' '.join(tokens)
