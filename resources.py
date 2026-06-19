# resources.py — Cached models and database client
# All heavy objects are loaded once and reused

import streamlit as st
import chromadb

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from config import EMBED_MODEL, OLLAMA_MODEL, CHROMA_PATH


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm():
    return ChatOllama(model=OLLAMA_MODEL)


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)
