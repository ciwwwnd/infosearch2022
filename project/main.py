from transformers import AutoModel, AutoTokenizer
from scipy import sparse
import numpy as np
import pickle
import torch
from preprocess import preprocess


answers, questions  = np.load('model/answers.npy', allow_pickle=True), np.load('model/questions.npy', allow_pickle=True)

q_bm25, q_tfidf, q_bert = sparse.load_npz('model/questions_bm25.npz'), sparse.load_npz('model/questions_tfidf.npz'), sparse.load_npz('model/bertmatrix.npz')

with open('model/TfIdf_Vectorizer.pk', 'rb') as fp:
    vectorizer = pickle.load(fp)

model, tokenizer = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru'), AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')

def query_index(query, vectorization_method):
    if vectorization_method == 'bert':
        t = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return sparse.csr_matrix(embeddings[0].cpu().numpy())
    elif vectorization_method == 'bm25' or vectorization_method == 'tfidf':
        preprocessed_query = preprocess(query)
        query_vector = vectorizer.transform([preprocessed_query])
        return query_vector

def search(query, vectorization_method):
    matrices = {'bm25':  q_bm25,
                'tfidf': q_tfidf,
                'bert':  q_bert}

    query = query_index(query, vectorization_method)
    res= np.dot(matrices[vectorization_method], query.T).toarray()
    scores = np.argsort(res, axis=0)[::-1][:5]
    return [(answers[i], questions[i]) for i in scores.ravel()]
