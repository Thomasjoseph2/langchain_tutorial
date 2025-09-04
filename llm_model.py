import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm=GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)

prompt="what is the capital of india?"

response=llm.invoke(prompt)
print(response)