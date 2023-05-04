import os
import pandas as pd

import streamlit as st
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

prompt_template = """
Please answer this question succinctly and professionally:
{query}

If you don't know the answer, just reply: not available.
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=prompt_template
)

def load_pandas_agent():
    chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
    df = pd.read_csv("data/sales_data.csv")
    agent = create_pandas_dataframe_agent(chat, df, verbose=False)
    return agent

def load_chained_agent():
    llm = OpenAI(temperature=0)
    serpapi_api_key=os.getenv('SERPAPI_API_KEY')
    toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
    agent = initialize_agent(toolkit, 
                             llm, 
                             agent="zero-shot-react-description", 
                             verbose=False, 
                             return_intermediate_steps=True)
    return agent
    
try:
    prompt_formatted = prompt.format(query="What is the most beautiful color?")
    agent = load_chained_agent()
    response = agent({"input": prompt_formatted})
    print(response["output"])
except Exception as e:
    print(e)
