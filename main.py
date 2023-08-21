"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def load_chroma():
    embed_model = OpenAIEmbeddings()
    db = Chroma(persist_directory="./chromadb", embedding_function=embed_model)
    return db

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    db = load_chroma()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat Demo", page_icon=":robot:")
st.header("Chat Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hi, I'm here to ask some questions about Wazuh!", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(query=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
