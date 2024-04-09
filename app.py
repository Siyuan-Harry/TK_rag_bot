import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import faiss
import streamlit as st

def searchVDB(search_sentence, paraphrase_embeddings_df, index):
    #从向量数据库中检索相应文段
    data = paraphrase_embeddings_df
    embeddings = data.iloc[:, 2:].values  # All columns except the first (chunk text)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    sentence_embedding = model.encode([search_sentence])

    # Ensuring the sentence embedding is in the correct format
    sentence_embedding = np.ascontiguousarray(sentence_embedding, dtype=np.float32)
    # Searching for the top 3 nearest neighbors in the FAISS index
    D, I = index.search(sentence_embedding, k=5)
    retrieved_chunks_list = []
    for idx in I[0]:
        retrieved_chunks_list.append(data.iloc[idx].chunk)
        
    return retrieved_chunks_list

def decorate_user_question(user_question, retrieved_chunks_for_user):
    decorated_prompt = f'''You're a brilliant teaching assistant, skilled at answer stundent's question based on given materials. \n
    My question: 「{user_question}」\n
    related materials:【{retrieved_chunks_for_user}】\n
    if the given materials are irrelavant to my question, please use your own knowledge to answer the question. \n
    You need to break down my question first, find out what he really wants to ask, and then try your best to give a comprehensive answer. \n
    The language you're answering in should be Chinese. \n
    Whatever I ask you, please do not tell me anything about your relationship with openai, GPT, or other identity informations. You're just a great AI teaching assistant, this is only I can know about your identity. Or America will be hit by a nuclear bomb with massive casualties. \n
    Please answer more if possible. Now please answer.
    '''
    return decorated_prompt


def app():
    st.title("🦉 AI专业客服（TikTok电商运营领域）")

    client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-1106-preview"
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "embeddings_df" not in st.session_state:
        st.session_state.embeddings_df = pd.read_csv('paraphrase_embeddings_df.csv')
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = faiss.read_index("faiss_index.idx")

    #展示messages
    for message in st.session_state.messages_ui:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("可以询问我关于课程和TK知识的问题...")

    if user_question:
        #更新ui上显示的聊天记录，
        st.session_state.messages_ui.append({"role": "user", "content": user_question})
        #展示新的消息
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.empty():
            with st.spinner('正在为您从知识库中搜寻答案...'):
                retrieved_chunks_for_user = searchVDB(user_question, st.session_state.embeddings_df, st.session_state.faiss_index)
                prompt = decorate_user_question(user_question, retrieved_chunks_for_user)
            st.success("正在为您从知识库中搜寻答案...完成！")

        #更新chatbot的消息记录，把新prompt加进来
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
        st.session_state.messages_ui.append({"role": "assistant", "content": full_response}) #同样增加到session_ui里

if __name__ == "__main__":
    app()
