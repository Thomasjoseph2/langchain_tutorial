from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize two different models
# gemini-2.5-flash is typically faster and more cost-effective for simple tasks
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6, api_key=api_key)

# gemini-2.5-pro is a more powerful model for complex tasks
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=api_key)

llm_translate = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)

# --- Define Prompts using LCEL ---

# Prompt template to generate a question
# Note: We now use ChatPromptTemplate.from_template
prompt_question = ChatPromptTemplate.from_template(
    "You are a creative assistant. Generate a single, simple, and interesting question about the topic: '{topic}'. " \
    "The question should be suitable for a 5-year-old."
)

# Prompt template to answer the question
# The input variable is inferred from the template string as 'question'
prompt_answer = ChatPromptTemplate.from_template(
    "You are a knowledgeable professor. Provide a clear and simple answer to the following question: '{question}'.return the question and the answer both"
)

promp_translate = ChatPromptTemplate.from_template("Translate the following English text to French: '{text}'.Return the text both in English and French")
# --- Build and Combine Chains using LCEL Piping ---

# The overall chain is now built by piping components together.
# 1. The input dictionary `{"topic": "..."}` is passed to `prompt_question`.
# 2. The formatted prompt is passed to the `llm_flash` model.
# 3. The model's output message is converted to a string by `StrOutputParser`.
# 4. The resulting string (the question) is passed to `prompt_answer`.
# 5. The newly formatted prompt is passed to the `llm_pro` model.
# 6. The final output message is converted to a string.

overall_chain = (
    prompt_question
    | llm_flash
    | StrOutputParser()
    | prompt_answer
    | llm_pro
    | StrOutputParser()
    | promp_translate
    | llm_translate
    | StrOutputParser()
)


# --- Run the chain ---
topic = "the sun"
print(f"--- Running Chain for topic: {topic} ---")

# With LCEL, you use .invoke()
# The input is a dictionary matching the input variable of the *first* prompt.
response = overall_chain.invoke({"topic": topic})

print("\n--- Final Answer ---")
print(response)