import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

def load(): 
    '''Load documents from /data'''
    loader = TextLoader('data/test.txt')
    return loader.load()

def split(docs):
    '''Split docs into chunks'''
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def store(docs):
    '''Store embeddeds in vector store (chromadb)'''
    embed_model = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embed_model, persist_directory="./chromadb")
    return vector_store

def main():
    docs = load()
    chunks = split(docs)
    store(chunks)
    pass

if __name__ == '__main__':
    main()
