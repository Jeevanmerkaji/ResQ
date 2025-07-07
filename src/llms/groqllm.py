from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

class GroqLLM:
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        try:
            print(os.getenv("GROQ_API_KEY"))
            os.environ["GROQ_API_KEY"] = self.groq_llm_key =  os.getenv("GROQ_API_KEY")
            llm = ChatGroq(api_key=self.groq_llm_key, model="mistral-7b-instruct")
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with the Exception  {e}")
        
        