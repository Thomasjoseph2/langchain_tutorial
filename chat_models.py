import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)
history=[]

print("chat bot initialized....")
print("type 'exit' to quit")

while True:
    user_input=input("you: ")
    if user_input.lower()=='exit':
        print("exiting chat...")
        break
    history.append(HumanMessage(content=user_input))
    response=llm.invoke(history)
    ai_message=response.content
    history.append(AIMessage(content=ai_message))
    
    print(F"AI :{ai_message}")
    
