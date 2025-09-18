from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_pet_name(animal_type: str, pet_color):
    llm = OllamaLLM(model="llama3")

    prompt_template_name = PromptTemplate(
        input_variables=["animal_type","pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest me five cool names."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    response = name_chain.invoke({"animal_type": animal_type, "pet_color": pet_color})
    return response

if __name__ == "__main__":
    print(generate_pet_name("cow", "white"))