##############################################################################
# Utility methods for building LLMs and agent models
#
# @philmui
# Mon May 1 18:34:45 PDT 2023
##############################################################################

import os
import pandas as pd

import streamlit as st
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import SQLDatabase, SQLDatabaseChain


def load_chat_agent(verbose=True):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    return chat

def load_sales_agent(verbose=True):
    '''
    Hard-coded agent that gates an internal sales CSV file for demo
    '''
    chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
    df = pd.read_csv("data/sales_data.csv")
    agent = create_pandas_dataframe_agent(chat, df, verbose=verbose)
    return agent

def load_sqlite_agent(model_name="text-davinci-003"):
    '''
    Hard-coded agent that gates a sqlite DB of digital media for demo
    '''
    llm = OpenAI(model_name=model_name, temperature=0)
    sqlite_db_path = "./data/Chinook_Sqlite.sqlite"
    db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    return db_chain

def load_chained_agent(verbose=True, model_name="text-davinci-003"):
    llm = OpenAI(model_name=model_name, temperature=0)
    toolkit = load_tools(["serpapi", "open-meteo-api", "news-api", 
                          "python_repl", "wolfram-alpha", 
                          "pal-math", "pal-colored-objects"],  # "tmdb-api"], 
                            llm=llm, 
                            serpapi_api_key=os.getenv('SERPAPI_API_KEY'),
                            news_api_key=os.getenv('NEWS_API_KEY'),
                            tmdb_bearer_token=os.getenv('TMDB_BEARER_TOKEN')
                            )
    agent = initialize_agent(toolkit, 
                             llm, 
                             agent="zero-shot-react-description", 
                             verbose=verbose, 
                             return_intermediate_steps=True)
    return agent

