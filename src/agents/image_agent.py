import base64
from langchain_core.messages import HumanMessage

from src.llm.gemini_llm import get_gemini_llm


def encode_image(image_path):
    """
    Convert image file into base64 string for Gemini.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def analyze_image(image_path, question):
    """
    Analyze an uploaded image using Gemini Vision.
    """

    llm = get_gemini_llm()

    image_base64 = encode_image(image_path)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            },
        ]
    )

    response = llm.invoke([message])

    return response.content