import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import faiss
import streamlit as st

@st.cache_data
def get_vdb():
    embeddings_df = pd.read_csv('paraphrase_embeddings_df.csv')
    faiss_index = faiss.read_index("faiss_index.idx")
    return embeddings_df, faiss_index

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

def decorate_user_question(user_question, retrieved_chunks_for_user):
    decorated_prompt = f'''You're a brilliant teaching assistant, skilled at answer stundent's question based on given materials.
    student's question: 「{user_question}」
    related materials:【{retrieved_chunks_for_user}】
    if the given materials are irrelavant to student's question, please use your own knowledge to answer the question.
    You need to break down the student's question first, find out what he really wants to ask, and then try your best to give a comprehensive answer.
    The language you're answering in should aligned with what student is using.
    Now I'm the student, please answer it for me.
    '''
    return decorated_prompt



def app():
    st.title("💡AI助教")

    client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-1106-preview"
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #展示messages
    for message in st.session_state.messages_ui:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("可以询问我关于课程和TK知识的问题...")

    if user_question:
        #更新ui上显示的聊天记录
        st.session_state.messages_ui.append({"role": "user", "content": user_question})

        #展示新的消息
        with st.chat_message("user"):
            st.markdown(user_question)
        
        embeddings_df, faiss_index = get_vdb()
        retrieved_chunks_for_user = searchVDB(user_question, embeddings_df, faiss_index)
        prompt = decorate_user_question(user_question, retrieved_chunks_for_user)

        #更新chatbot的消息记录，把新消息加进来
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages #用chatbot那边的隐藏消息记录
                ],
                stream=True,
            ):
                try:
                    full_response += response.choices[0].delta.content
                except:
                    full_response += ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response}) #再把新增的完整消息记录增加到session里

if __name__ == "__main__":
    app()