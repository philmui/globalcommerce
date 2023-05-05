
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from models import load_chat_agent, load_chained_agent, load_sales_agent, load_sqlite_agent

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
    ResponseSchema(name="artist", description="The name of the musical artist"),
    ResponseSchema(name="song", description="The name of the song that the artist plays")
]

# The parser that will look for the LLM output in my schema and return it back to me
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

chat_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("Given a command from the user, extract the artist and song names \n \
                                                    {format_instructions}\n{user_prompt}")  
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)


def chatAgent(chat_message):
    try:
        agent = load_chat_agent(verbose=True)
        output = agent([HumanMessage(content=chat_message)])
    except:
        output = ""
    return output


def instructAgent(question_text, model_name):
    instruction = instruct_prompt.format(query=question_text)
    output = ""
    try:
        agent = load_chained_agent(verbose=True, model_name=model_name)
        response = agent({"input": instruction})
        if response is None or "not available" in response["output"]:
            response = ""
        else:
            output = response["output"]
    except: 
        output = ""

    if len(output) < 12: 
        output = salesAgent(instruction)
    return output


def salesAgent(instruction):
    output = ""
    try:
        agent = load_sales_agent(verbose=True)
        output = agent.run(instruction)
        print("panda> " + output)
    except:
        output = "Sorry: no response possible right now"
    return output

def chinookAgent(instruction, model_name):
    output = ""
    try:
        agent = load_sqlite_agent(model_name)
        output = agent.run(instruction)
        print("chinook> " + output)
    except:
        output = "Sorry: no response possible right now"
    return output