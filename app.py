##############################################################################
# Main script that builds the UI & connects the logic for an LLM-driven
# query frontend to a "Global Commerce" demo app.
#
# @philmui
# Mon May 1 18:34:45 PDT 2023
##############################################################################


import streamlit as st
from pprint import pprint
from agents import agentController, salesAgent, chinookAgent, chatAgent

##############################################################################

st.set_page_config(page_title="Global", 
                   page_icon=":cart:", 
                   layout="wide")
st.header("📦 Global 🛍️")

col1, col2 = st.columns([1,1])

with col1:
    option_llm = st.selectbox(
        "Model",
        ('text-davinci-003', 
         'text-babbage-001', 
         'text-curie-001', 
         'text-ada-001',
         'gpt-4',
         'gpt-3.5-turbo',
         'google/flan-t5-xl',
         'databricks/dolly-v2-3b',
         'bigscience/bloom-1b7')
    )
with col2:
    option_mode = st.selectbox(
        "LLM mode",
        ("Instruct (all)",
         "Chat (high temperature)",
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
if question_text and len(question_text) > 1:
    output=""
    if option_mode == "Internal-Sales":
        output = salesAgent(question_text)
    elif option_mode == "Internal-Merchant":
        output = chinookAgent(question_text, option_llm)
    elif option_mode.startswith("Chat"):
        response = chatAgent(question_text)
        if response and response.content:
            output = response.content
        else:
            output = response
    else:
        output = agentController(question_text, option_llm)

    height = min(2*len(output), 280)
    st.text_area(label="In response ...", 
                 value=output, height=height)
    # st.code(pprint(output), language="markdown")

##############################################################################

st.markdown(
    """
    <style>
    textarea[aria-label^="ex"] {
            font-size: 0.8em !important;
            font-family: Arial, sans-serif !important;
            color: gray !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### 3 types of reasoning:")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.markdown("__Common sense reasoning__")
    st.text_area(label="ex1", label_visibility="collapsed", height=120,
                 value="🔹 Why is the sky blue?\n" +
                       "🔹 How to avoid touching a hot stove?\n" +
                       "🔹 Please give tips to win a 3200m track race?\n" +
                       "🔹 Please advise on how best to prepare for retirement?"
                       )

with col2:
    st.markdown("__Local ('secure') reasoning__")
    st.text_area(label="ex2", label_visibility="collapsed", height=120,
                 value="🔹 For my company, what is the total sales " +
                       "broken down by month?\n" +
                       "🔹 How many total artists are there in each "+
                       "genres in our digital media database?")

with col3:
    st.markdown("__Enhanced reasoning__ [🎵](https://www.youtube.com/watch?v=hTTUaImgCyU&t=62s)")
    st.text_area(label="ex3", label_visibility="collapsed", height=120,
                 value="🔹 Who is the president of South Korea?  " +
                       "What is his favorite song? How old is he? " +
                       "What is the smallest prime greater than his age?\n" +
                       "🔹 What is the derivative of f(x)=3*log(x)*sin(x)?")

st.image(image="images/plugins.png", width=700, caption="salesforce.com")
st.image(image="images/chinook.png", width=420, caption="Digital Media Schema")

##############################################################################
