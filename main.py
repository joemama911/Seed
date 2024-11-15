import os
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from psi import llm_automation, Linkedin_post
from openai import OpenAI
from langchain_openai import ChatOpenAI
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import requests
from dotenv import load_dotenv
import kaggle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings
from langchain.agents import Tool, AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import nomic
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
import shutil

import os
import sqlite3
from openai import OpenAI
import streamlit




os.environ['COHERE_API_KEY'] = "yTmKdlP6vaGOZ91YAlPCKqMUpvmD2rgSoZqZJRHS"
os.environ['TAVILY_API_KEY'] = "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"
nomic.cli.login("nk-TbdtpiqAFh3TRTPDLItfr6FLiUpXYb2TwapWvrEhi_g")



chat = ChatCohere(model="command-r-plus", temperature=0.3)

internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."



class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")
internet_search.args_schema = TavilySearchInput


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="you can train deep learning models but you have to locate the datasets inside the file location shared with you , IF you face no such file then search the directory shared with you ",
    func=python_repl,
    args_schema=ToolInput

)

# from langchain_core.pydantic_v1 import BaseModel, Field
class ToolInput(BaseModel):
    code: str = Field(description="you can train deep learning models but you have to locate the datasets inside the file location shared with you , IF you face no such file then search the directory shared with you ")
repl_tool.args_schema = ToolInput

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3.1")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain 

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text +=page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3.1")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain 

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text +=page.extract_text()
    return text

'''docs=get_pdf_text("E:\Linkedin_Automation_with_Generative_AI\Trumio (1).pdf")

text_chunks = get_text_chunks(docs)
                

vectorstore = get_vectorstore(text_chunks)'''


def one(prompt):
        docs=get_pdf_text("E:\Linkedin_Automation_with_Generative_AI\Trumio (1).pdf")
        text_chunks = get_text_chunks(docs)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)
        response = conversation_chain(prompt)
        return response 

                
class rag:
    def rag_runner(prompt):
        response=one(prompt)
        return response
    
class ragger(BaseModel):
    code: str = Field(description="you are documentation agent which is supposed to read the document and summerize")

rag_tool=Tool(
    name="rag_tool",
    description="you are documentation agent which is supposed to read the document and summerize",
    func=rag.rag_runner,
    arg_schema=ragger
)

import kagglehub
class kag:
    def kaggler(link):
        path=kagglehub.dataset_download(link)
        return path


class duck(BaseModel):
    code : str = Field(description="you are a kaggle download tool , which uses the link provided to download the dataset and store it locally ")

kaggler=Tool(
    name="kaggler",
    description=("you are a kaggle download tool , which uses the link provided to download the dataset and store it locally "),
    func=kag.kaggler,
    arg_schema=duck
)

prompt = ChatPromptTemplate.from_template("{input}try again and again if you ever fail")

agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, repl_tool,rag_tool,kaggler],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool,rag_tool,kaggler], verbose=True)


hello = True  
while hello :
    inputt=input("wassup\n")
    ans=agent_executor.invoke({"input":inputt})
