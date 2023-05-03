import os
import pandas as pd

import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

prompt_template = """
Please answer this question succinctly and professionally:
{query}
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=prompt_template
)

def load_LLM():
    llm = OpenAI(model_name=option_llm, temperature=0)
    return llm

def load_pandas_agent():
    chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
    df = pd.read_csv("data/sales_data.csv")
    agent = create_pandas_dataframe_agent(chat, df, verbose=False)
    return agent

##############################################################################

st.set_page_config(page_title="Global Commerce", page_icon=":robot:")
st.header("Global Commerce")

col1, col2 = st.columns([1,2])

with col1:
    option_llm = st.selectbox(
        "Which LLM would you like to use?",
        ('text-davinci-003', 
         'text-babbage-001', 
         'text-ada-001',
         'cohere',
         'dolly')
    )
with col2:
    pass

def get_question():
    input_text = st.text_area(label="Your question ...", 
                              placeholder="Ask me here.",
                              key="question_text")
    return input_text

question_text = get_question()
if question_text:
    st.markdown(f"_chosen llm_: {option_llm}")
    prompt_formatted = prompt.format(query=question_text)
    # llm = load_LLM()
    # response = llm(prompt_formatted)
    agent = load_pandas_agent()
    response = agent.run(prompt_formatted)
    st.write(f"Your question: {question_text}")
    st.write(response)

##############################################################################

col1, col2 = st.columns([1,2])

with col1:
    st.markdown("#### 3 types of reasoning:")
    st.markdown("* LLM and common sense reasoning")
    st.markdown("* local ('secure') data analysis")
    st.markdown("* 3rd party enhanced reasoning")

with col2:
    st.image(image="images/plugins.png", width=500, caption="salesforce.com")

##############################################################################
