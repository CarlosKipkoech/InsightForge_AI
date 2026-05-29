import os
import streamlit as st

from src.agents.code_agent import analyze_code
from src.llm.groq_llm import get_groq_llm
from src.rag.pdf_loader import load_and_split_pdf
from src.rag.qdrant_store import create_qdrant_store
from src.rag.rag_chain import create_rag_answer
from src.agents.image_agent import analyze_image
from src.agents.supervisor import route_question


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 InsightForge AI")
st.caption("Multi-Agent Research & Document Intelligence Platform")


# =====================================================
# APPLICATION SETUP
# =====================================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None


# =====================================================
# SIDEBAR - KNOWLEDGE SOURCES
# =====================================================
with st.sidebar:

    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded_file:

        file_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Process Document"):

            with st.spinner("Processing document..."):

                chunks = load_and_split_pdf(file_path)

                st.session_state.vector_store = (
                    create_qdrant_store(chunks)
                )

                st.success(
                    f"Document processed successfully! "
                    f"Created {len(chunks)} chunks."
                )

    st.header("🖼️ Image Upload")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image_path = os.path.join(
            UPLOAD_DIR,
            uploaded_image.name
        )

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.session_state.image_path = image_path

        st.image(
            image_path,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.success("Image uploaded successfully!")


# =====================================================
# LOAD LANGUAGE MODEL
# =====================================================
llm = get_groq_llm()


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# CHAT INPUT
# =====================================================
prompt = st.chat_input(
    "Ask InsightForge AI..."
)

if prompt:

    # ---------------------------------------------
    # DISPLAY USER MESSAGE
    # ---------------------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------------------------
    # GENERATE RESPONSE USING SUPERVISOR ROUTING
    # ---------------------------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            route = route_question(
                question=prompt,
                has_pdf=st.session_state.vector_store is not None,
                has_image=st.session_state.image_path is not None
            )

            st.caption(f"🔀 Route Selected: {route}")

            # ============================================
            # IMAGE AGENT
            # ============================================
            if (
                route == "image"
                and st.session_state.image_path is not None
            ):

                answer = analyze_image(
                    image_path=st.session_state.image_path,
                    question=prompt
                )

            # ============================================
            # DOCUMENT AGENT
            # ============================================
            elif (
                route == "document"
                and st.session_state.vector_store is not None
            ):

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        prompt,
                        k=4
                    )
                )

                answer = create_rag_answer(
                    question=prompt,
                    docs=docs
                )

            # ============================================
            # CODE AGENT PLACEHOLDER
            # ============================================
            elif route == "code":
                answer = analyze_code(
                    question=prompt
                )

            # ============================================
            # GENERAL AGENT
            # ============================================
            else:

                response = llm.invoke(prompt)

                answer = response.content

            st.markdown(answer)

    # ---------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # ---------------------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )