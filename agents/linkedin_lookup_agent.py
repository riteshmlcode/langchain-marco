import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily

load_dotenv()

def lookup(name:str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temparature=0)
    template = """given the full name {name_of_person} I want you to get it me a link to thier
                linkedin page. Your answer should only contain a URL"""
    prompt_template = PromptTemplate(input_variables=["name_of_person"],template=template)
    tools_for_agent = [
        Tool(
            name="Crawl Google for Linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need to get the Linkedin profile page URL",
            args=prompt_template.format(name_of_person=name)
        )
    ]
    
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True
    )

    result = agent_executor.invoke(
        input={"input":prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]
    return linkedin_profile_url

if __name__ == "__main__":
    linkedin_url = lookup(name="Eden Marco")
    print(linkedin_url)


    



