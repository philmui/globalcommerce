##############################################################################
# Agent interfaces that bridges private capability agents (pandas, 
# sql, ...), 3rd party plugin agents (search, weather, movie, ...),
# and 3rd party LLMs
#
# @philmui
# Mon May 1 18:34:45 PDT 2023
##############################################################################

from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, \
                              HumanMessagePromptTemplate
from models import load_chat_agent, load_chained_agent, load_sales_agent, \
                   load_sqlite_agent

import logging

logger = logging.getLogger(__name__)

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

instruct_template = """
Please answer this question clearly with easy to follow reasoning:
{query}

If you don't know the answer, just reply: not available.
"""

instruct_prompt = PromptTemplate(
    input_variables=["query"],
    template=instruct_template
)

response_schemas = [
    ResponseSchema(name="artist", 
                   description="The name of the musical artist"),
    ResponseSchema(name="song", 
                   description="The name of the song that the artist plays")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

LOCAL_MAGIC_TOKENS = ["my company", "for us", "our company", "our sales"]
DIGITAL_MAGIC_TOKENS = ["digital media", "our database", "our digital"]

def is_magic(sentence, magic_tokens):
    return any([t in sentence.lower() for t in magic_tokens])


chat_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            "Given a command from the user, extract the artist and \
             song names \n{format_instructions}\n{user_prompt}")  
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)


def chatAgent(chat_message):
    try:
        agent = load_chat_agent(verbose=True)
        output = agent([HumanMessage(content=chat_message)])
    except:
        output = "Please rephrase and try chat again."
    return output


def agentController(question_text, model_name):
    output = ""

    if is_magic(question_text, LOCAL_MAGIC_TOKENS):
        output = salesAgent(question_text)
        print(f"🔹 salesAgent: {output}")
    elif is_magic(question_text, DIGITAL_MAGIC_TOKENS):
        output = chinookAgent(question_text, model_name)
        print(f"🔹 chinookAgent: {output}")
    else:
        try:
            instruction = instruct_prompt.format(query=question_text)
            logger.info(f"instruction: {instruction}")
            agent = load_chained_agent(verbose=True, model_name=model_name)
            response = agent([instruction])
            if response is None or "not available" in response["output"]:
                response = ""
            else:
                output = response['output']
                logger.info(f"🔹 Steps: {response['intermediate_steps']}")
        except Exception as e: 
            output = "Most likely ran out of tokens ..."
            logger.error(e)

    return output


def salesAgent(instruction):
    output = ""
    try:
        agent = load_sales_agent(verbose=True)
        output = agent.run(instruction)
        print("panda> " + output)
    except Exception as e:
        logger.error(e)
        output = f"Rephrasing your prompt could get better sales results {e}"
    return output

def chinookAgent(instruction, model_name):
    output = ""
    try:
        agent = load_sqlite_agent(model_name)
        output = agent.run(instruction)
        print("chinook> " + output)
    except Exception as e:
        logger.error(e)
        output = "Rephrasing your prompt could get better db results {e}"
    return output