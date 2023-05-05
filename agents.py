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

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

instruct_template = """
Please answer this question succinctly and professionally:
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


def instructAgent(question_text, model_name):
    instruction = instruct_prompt.format(query=question_text)
    output = ""

    if is_magic(question_text, LOCAL_MAGIC_TOKENS):
        output = salesAgent(instruction)
    elif is_magic(question_text, DIGITAL_MAGIC_TOKENS):
        output = chinookAgent(question_text, model_name)
    else:
        try:
            agent = load_chained_agent(verbose=True, model_name=model_name)
            response = agent({"input": instruction})
            if response is None or "not available" in response["output"]:
                response = ""
            else:
                output = response["output"]
        except: 
            output = "Please rephrase and try again ..."

    return output


def salesAgent(instruction):
    output = ""
    try:
        agent = load_sales_agent(verbose=True)
        output = agent.run(instruction)
        print("panda> " + output)
    except:
        output = "Please rephrase and try again for company sales data"
    return output

def chinookAgent(instruction, model_name):
    output = ""
    try:
        agent = load_sqlite_agent(model_name)
        output = agent.run(instruction)
        print("chinook> " + output)
    except:
        output = "Please rephrase and try again for digital media data"
    return output