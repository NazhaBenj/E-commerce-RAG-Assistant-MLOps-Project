import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.rag.rag_chain import RAGChain


st.set_page_config(
    page_title="Jumia RAG Assistant",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Jumia Maroc - RAG Assistant")
st.caption("Assistant intelligent basé sur la documentation officielle de Jumia Maroc.")


@st.cache_resource
def load_rag_chain():
    return RAGChain(top_k=3)


try:
    rag_chain = load_rag_chain()
except Exception as error:
    st.error(f"Erreur lors du chargement du système RAG : {error}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("À propos")
    st.write("Assistant conversationnel basé sur une architecture RAG.")
    st.markdown("**Technologies :**")
    st.write("- Sentence Transformers")
    st.write("- ChromaDB")
    st.write("- BM25 + RRF")
    st.write("- Ollama / Llama 3.2")

    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources documentaires"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['document_id']}** - {source['topic']}"
                    )
                    if source.get("source_url"):
                        st.markdown(
                            f"[Consulter le document]({source['source_url']})"
                        )


question = st.chat_input("Posez votre question sur Jumia Maroc...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                result = rag_chain.invoke(question)
                st.markdown(result["answer"])

                with st.expander("Sources documentaires"):
                    for source in result["sources"]:
                        st.markdown(
                            f"**{source['document_id']}** - {source['topic']}"
                        )
                        if source.get("source_url"):
                            st.markdown(
                                f"[Consulter le document]({source['source_url']})"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception as error:
                st.error(f"Erreur : {error}")
