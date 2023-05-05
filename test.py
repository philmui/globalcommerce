from models import load_chained_agent
from agents import chatAgent

print(chatAgent("why is the sky blue?"))
# try:
#     prompt_formatted = prompt.format(query="""
#     Who is the president of South Korea?  What is his age?  What is the digit sum of his age?
#     """)
#     agent = load_chained_agent(verbose=True)
#     response = agent({"input": prompt_formatted})
#     print(response["output"])
# except Exception as e:
#     print(e)
