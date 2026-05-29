from src.llm.groq_llm import get_groq_llm


def analyze_code(question: str):
    """
    Code Agent:
    Handles programming, debugging, explaining code,
    improving code, and suggesting architecture.
    """

    llm = get_groq_llm()

    prompt = f"""
You are the Code Agent for InsightForge AI.

Your role:
- Explain code clearly
- Debug errors
- Suggest improvements
- Generate clean Python code
- Help with Streamlit, LangChain, LangGraph, RAG, and AI apps

Rules:
- Be practical
- Explain in simple language
- Give complete code when needed
- Avoid unnecessary theory

User request:
{question}
"""

    response = llm.invoke(prompt)

    return response.content