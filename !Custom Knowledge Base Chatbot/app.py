import streamlit as st
import json
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from streamlit_chat import message

# Utility functions
from utils import load_sentence_model, load_qa_pipeline, load_generative_pipeline, compute_embeddings, initialize_faiss_index, retrieve_relevant_documents, generate_answer, log_chat

# Initialize models
sentence_model = load_sentence_model()
qa_pipeline = load_qa_pipeline()
generative_pipeline = load_generative_pipeline()

# Streamlit UI
st.title("Custom Knowledge Base Chatbot")

if 'document_embeddings' not in st.session_state:
    st.session_state['document_embeddings'] = None
    st.session_state['docs'] = None
    st.session_state['questions'] = []
    st.session_state['answers'] = []
    st.session_state['faiss_index'] = None

urls = st.text_input("Enter document URLs (comma-separated):")
if urls:
    try:
        url_list = [url.strip() for url in urls.split(',')]
        loader = UnstructuredURLLoader(urls=url_list)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        docs = [doc.page_content for doc in split_docs]
        document_embeddings = compute_embeddings(docs, sentence_model)
        faiss_index = initialize_faiss_index(document_embeddings)
        st.session_state['docs'] = docs
        st.session_state['document_embeddings'] = document_embeddings
        st.session_state['faiss_index'] = faiss_index
        st.success("Documents loaded and processed.")
    except Exception as e:
        st.error(f"Error loading documents: {e}")

mode = st.radio("Choose mode:", ("QA", "Generative"))

question = st.text_input("Ask a question:")
if question:
    st.session_state['questions'].append(question)
    if st.session_state['faiss_index'] and st.session_state['docs']:
        relevant_docs = retrieve_relevant_documents(question, st.session_state['faiss_index'], st.session_state['docs'], sentence_model, top_k=2)
        if relevant_docs:
            answer = generate_answer(question, relevant_docs, mode, qa_pipeline, generative_pipeline)
        else:
            answer = "No relevant documents found. Generating answer..."
            answer += generate_answer(question, [], "Generative", qa_pipeline, generative_pipeline)
    else:
        # If no documents are loaded, use generative model directly
        answer = generate_answer(question, [], mode, qa_pipeline, generative_pipeline)
    
    st.session_state['answers'].append(answer)
    log_chat(question, answer)
    message(question)
    message(answer, is_user=True)

# Display recent chat history and allow expansion to view entire history
if st.session_state['questions']:
    with st.expander("Chat History"):
        for q, a in zip(st.session_state['questions'], st.session_state['answers']):
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
    if len(st.session_state['questions']) > 0:
        st.write("## Recent Chat")
        st.write(f"**Q:** {st.session_state['questions'][-1]}")
        st.write(f"**A:** {st.session_state['answers'][-1]}")
