from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    print("Hello Langchain")
    load_dotenv()
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temparature=0)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    chain = summary_prompt_template | model

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True
    )
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
