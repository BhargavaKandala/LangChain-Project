from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


def generate_pet_name(animal_type: str, pet_color):
    llm = OllamaLLM(model="llama3")

    prompt_template_name = PromptTemplate(
        input_variables=["animal_type","pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest me five cool names."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
    response = name_chain.invoke({"animal_type": animal_type, "pet_color": pet_color})
    return response

def langchain_agent():
    llm = OllamaLLM(model="llama3")
    
    tools = load_tools(["wikipedia", "llm-math"], llm = llm)
    
    agent = initialize_agent(
        tools, llm, agent_type="zero-shot-react-description", verbose=True
    )

    result = agent.invoke(
        "What is the average age of a dog? Multiply the age by 3"
    )
    print(result)
    
if __name__ == "__main__":
    langchain_agent()
    # print(generate_pet_name("cow", "white"))