from src.llm.groq_llm import get_groq_llm


def route_question(question: str, has_pdf:bool,has_image:bool):
    llm = get_groq_llm()

    prompt = f"""
You are the Supervisor Agent for InsightForge AI.

Your job is to choose the best route for the user's request.

Available routes:
- document: use when the question is about an uploaded PDF/document
- image: use when the question is about an uploaded image/screenshot/diagram
- code: use when the question is about programming, debugging, or explaining code
- general: use for normal questions

Current app state:
PDF uploaded: {has_pdf}
Image uploaded: {has_image}

User question:
{question}

Return only one word:
document, image, code, or general
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    if route not in ["document", "image", "code", "general"]:
        route = "general"

    return route