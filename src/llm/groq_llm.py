import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq


load_dotenv()


def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Check your .env file.")



    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )