import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import streamlit as st


def searchVDB(search_sentence, paraphrase_embeddings_df, index):
    #从向量数据库中检索相应文段
    try:
        data = paraphrase_embeddings_df
        embeddings = data.iloc[:, 1:].values  # All columns except the first (chunk text)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        model = SentenceTransformer('paraphrase-mpnet-base-v2')
        sentence_embedding = model.encode([search_sentence])

        # Ensuring the sentence embedding is in the correct format
        sentence_embedding = np.ascontiguousarray(sentence_embedding, dtype=np.float32)
        # Searching for the top 3 nearest neighbors in the FAISS index
        D, I = index.search(sentence_embedding, k=3)
        # Printing the top 3 most similar text chunks
        retrieved_chunks_list = []
        for idx in I[0]:
            retrieved_chunks_list.append(data.iloc[idx].chunk)

    except Exception:
        retrieved_chunks_list = []
        
    return retrieved_chunks_list


st.title("💡AI助教")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})