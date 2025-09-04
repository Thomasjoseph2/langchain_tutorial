from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=os.getenv("GEMINI_API_KEY"))

template="write a {tone} email to {company} about my interest in the {position} by highlighting my {skills}"

prompt_template=ChatPromptTemplate.from_template(template)

prompt=prompt_template.invoke({"tone":"energetic", "company":"samsung", 
                               "position":"Software Engineer",
                               "skills":"cloud computing, distributed systems, and problem-solving"})


response=llm.invoke(prompt)

print(response.content)