##############################################################################
# Main script that builds the UI & connects the logic for an LLM-driven
# query frontend to a "Global Commerce" demo app.
#
# @philmui
# Mon May 1 18:34:45 PDT 2023
##############################################################################


import streamlit as st
from agents import instructAgent, salesAgent, chinookAgent, chatAgent

##############################################################################

st.set_page_config(page_title="Global Commerce", page_icon=":robot:")
st.header("📦 Global Commerce 🛍️")

col1, col2 = st.columns([1,1])

with col1:
    option_llm = st.selectbox(
        "Model",
        ('text-davinci-003', 
         'text-babbage-001', 
         'text-ada-001',
         'gpt-4',
         'gpt-3.5-turbo',
         'cohere',
         'dolly')
    )
with col2:
    option_mode = st.selectbox(
        "LLM mode",
        ("Instruct",
         "Chat",
         "Wolfram-Alpha",
         "Internal-Sales",
         "Internal-Merchant"
         )
    )

def get_question():
    input_text = st.text_area(label="Your question ...", 
                              placeholder="Ask me anything ...",
                              key="question_text", label_visibility="collapsed")
    return input_text

question_text = get_question()
if question_text:
    output=""
    if option_mode == "Internal-Sales":
        output = salesAgent(question_text)
    elif option_mode == "Internal-Merchant":
        output = chinookAgent(question_text, option_llm)
    elif option_mode == "Chat":
        output = chatAgent(question_text).content
    else:
        output = instructAgent(question_text, option_llm)

    height = min(2*len(output), 280)
    st.text_area(label="In response ...", 
                 value=output, height=height)

##############################################################################

st.markdown("#### 3 types of reasoning:")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.markdown("__Common sense reasoning__")
    st.text_area(label="o1", label_visibility="collapsed",
                 value="> Why is the sky blue?\n" +
                       "> How to avoid touching a hot stove?")

with col2:
    st.markdown("__Local ('secure') analysis__")
    st.text_area(label="o2", label_visibility="collapsed",
                 value="> What is our total sales per month?")

with col3:
    st.markdown("__Enhanced reasoning__")
    st.text_area(label="o3", label_visibility="collapsed",
                 value="> Who is the president of South Korea?  " +
                       "What is his favorite song?  " +
                       "What is the smallest prime greater than his age?")

st.image(image="images/plugins.png", width=700, caption="salesforce.com")
st.image(image="images/chinook.png", width=700, caption="Chinook Digital Media")

##############################################################################
