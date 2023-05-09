##############################################################################
# Utility methods for building LLMs and agent models
#
# @philmui
# Mon May 1 18:34:45 PDT 2023
##############################################################################

import os
import pandas as pd

from langchain.agents import AgentType, load_tools, initialize_agent,\
                            create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import SQLDatabase, SQLDatabaseChain, HuggingFaceHub

OPENAI_LLMS = [ 
    'text-davinci-003', 
    'text-babbage-001', 
    'text-curie-001', 
    'text-ada-001'
]

OPENAI_CHAT_LLMS = [
    'gpt-3.5-turbo',     
    'gpt-4',
]

HUGGINGFACE_LLMS = [
    'google/flan-t5-xl',
    'databricks/dolly-v2-3b',
    'bigscience/bloom-1b7'
]

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def createLLM(model_name="text-davinci-003", temperature=0):
    llm = None
    if model_name in OPENAI_LLMS:
        llm = OpenAI(model_name=model_name, temperature=temperature)
    elif model_name in OPENAI_CHAT_LLMS:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    elif model_name in HUGGINGFACE_LLMS:
        llm = HuggingFaceHub(repo_id=model_name, 
                             model_kwargs={"temperature":1e-10})
    return llm


def load_chat_agent(verbose=True):
    return createLLM(OPENAI_CHAT_LLMS[0], temperature=0.5)

def load_sales_agent(verbose=True):
    '''
    Hard-coded agent that gates an internal sales CSV file for demo
    '''
    chat = createLLM(OPENAI_CHAT_LLMS[0], temperature=0.5)
    df = pd.read_csv("data/sales_data.csv")
    agent = create_pandas_dataframe_agent(chat, df, verbose=verbose)
    return agent

def load_sqlite_agent(model_name="text-davinci-003"):
    '''
    Hard-coded agent that gates a sqlite DB of digital media for demo
    '''
    llm = createLLM(OPENAI_LLMS[0])
    sqlite_db_path = "./data/Chinook_Sqlite.sqlite"
    db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    return db_chain

from langchain.tools import DuckDuckGoSearchRun, GoogleSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
def load_chained_agent(verbose=True, model_name="text-davinci-003"):
    llm = createLLM(model_name)
    toolkit = [DuckDuckGoSearchRun()]
    toolkit += load_tools(["serpapi", "open-meteo-api", "news-api", 
                           "python_repl", "wolfram-alpha"], 
                            llm=llm, 
                            serpapi_api_key=os.getenv('SERPAPI_API_KEY'),
                            news_api_key=os.getenv('NEWS_API_KEY'),
                            tmdb_bearer_token=os.getenv('TMDB_BEARER_TOKEN')
                            )

    agent = initialize_agent(toolkit, 
                             llm, 
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                             verbose=verbose, 
                             return_intermediate_steps=True)
    return agent