from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
# Load environment variables from .env
load_dotenv()

# Create a gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=os.getenv("GEMINI_API_KEY"))

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"animal": "zeebra", "fact_count": 5})

# Output
print(result)
