from langchain_core.prompts import ChatPromptTemplate

from src.llm.groq_llm import get_groq_llm

RAG_PROMPT = """
You are InsightForge AI, a document intelligence assistant.

Answer the user's question using only the context below.

If the answer is not in the context, say:
"I could not find that information in the uploaded document."

Context:
{context}

Question:
{question}
"""


def create_rag_answer(question:str,docs):
    llm = get_groq_llm()
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    chain =prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content