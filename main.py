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
    llm = OpenAI(model_name=option_llm, temperature=0)
    serpapi_api_key=os.getenv('SERPAPI_API_KEY')
    toolkit = load_tools(["serpapi", "wolfram-alpha"], 
                         llm=llm, 
                         serpapi_api_key=serpapi_api_key)
    agent = initialize_agent(toolkit, 
                             llm, 
                             agent="zero-shot-react-description", 
                             verbose=False, 
                             return_intermediate_steps=True)
    return agent

##############################################################################

st.set_page_config(page_title="Global Commerce", page_icon=":robot:")
st.header("Global Commerce")

col1, col2 = st.columns([1,1])

with col1:
    option_llm = st.selectbox(
        "Model",
        ('text-davinci-003', 
         'text-babbage-001', 
         'text-ada-001',
         'cohere',
         'dolly')
    )
with col2:
    option_mode = st.selectbox(
        "LLM mode",
        ("Instruct",
         "Chat",
         "Pandas",
         "Wolfram Alpha")
    )

def get_question():
    input_text = st.text_area(label="Your question ...", 
                              placeholder="Ask me here.",
                              key="question_text")
    return input_text

question_text = get_question()
if question_text:
    st.markdown(f"_chosen llm_: {option_llm}")
    prompt_formatted = prompt.format(query=question_text)
    output = ""

    try:
        agent = load_chained_agent()
        response = agent({"input": prompt_formatted})
        if response is None or "not available" in response["output"]:
            response = ""
        else:
            output = response["output"]
    except: 
        output = ""

    if len(output) < 12: 
        try:
            agent = load_pandas_agent()
            output = agent.run(prompt_formatted)
            print("==> " + output)
        except:
            output = "Sorry: no response possible right now"

    st.write(output)

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
