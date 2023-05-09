import os
from models import load_chained_agent
from agents import chatAgent
import langchain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# print(chatAgent("why is the sky blue?"))

# try:
#     prompt_formatted = prompt.format(query="""
#     Who is the president of South Korea?  What is his age?  What is the digit sum of his age?
#     """)
#     agent = load_chained_agent(verbose=True)
#     response = agent({"input": prompt_formatted})
#     print(response["output"])
# except Exception as e:
#     print(e)

from langchain.tools import DuckDuckGoSearchRun, GoogleSearchRun
from langchain.utilities import GoogleSearchAPIWrapper

def load_chained_agent(verbose=True, model_name="text-davinci-003"):
    llm = OpenAI(model_name=model_name, temperature=0)
    toolkit = [GoogleSearchRun(), DuckDuckGoSearchRun()]

    toolkit += load_tools(["open-meteo-api", "news-api", 
                          "python_repl", "wolfram-alpha", 
                          "pal-math", "pal-colored-objects"],  
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


PROMPT = "Who is the president of South Korea?  How old is he?  What is the smallest prime greater than his age?"

if __name__ == '__main__':
    agent = load_chained_agent()
    response = agent(PROMPT)
    if response is not None:
        """
        print("Steps: ")
        for action in response['intermediate_steps']:
            print()
            print(f"==> Tool: {action[0].tool}")
            print(f"    Input: {action[0].tool_input}")
            print(f"    Thought: {action[0].log}")
            print(f"    Finding: {action[1]}")
        """
        print(f"input: {response['input']}")
        print(f"output: {response['output']}")
