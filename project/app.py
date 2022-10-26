import streamlit as st
from main import search
import time
from argparse import ArgumentParser

start_time = time.time()
st.title("What do you want to know about love?")
query = st.text_input('Describe your problem, remember to ask all questions in Russian', key='6')
left_col, right_col = st.columns(2)
model_type = left_col.selectbox('Choose vectorization method', ['bert', 'bm25', 'tfidf'])

if st.button('Search'):
    
    search_results = search(query=query, vectorization_method=model_type)
    st.markdown(f'Время поиска: {round((time.time() - start_time), 3)} секунд')
    res = [reply[0] for reply in search_results]
    for rep in res:
        try:
            st.markdown(f'* {rep[0]}')
        except IndexError:
            pass