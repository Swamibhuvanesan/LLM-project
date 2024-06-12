import os
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline
import json
import warnings

# Set API Key
os.environ['HF_API_KEY'] = 'hf_' #replace with your Key.

# Initialize the sentence transformer model for embeddings
@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Faster model

# Initialize the QA pipeline with a smaller, faster model
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize the Generative pipeline for generating longer answers
@st.cache_resource(show_spinner=False)
def load_generative_pipeline():
    model_name = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to compute document embeddings in batches
def compute_embeddings(docs, sentence_model, batch_size=16):
    embeddings = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_embeddings = sentence_model.encode(batch_docs, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Function to initialize FAISS index
def initialize_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to retrieve relevant documents using FAISS
def retrieve_relevant_documents(question, faiss_index, documents, sentence_model, top_k=3):
    question_embedding = sentence_model.encode(question, convert_to_tensor=True).reshape(1, -1)
    distances, indices = faiss_index.search(question_embedding, top_k)
    return [documents[idx] for idx in indices[0]]

# Function to generate an answer based on the mode
def generate_answer(question, relevant_docs, mode, qa_pipeline, generative_pipeline):
    if mode == "QA" and relevant_docs:
        context = " ".join(relevant_docs)
        answer = qa_pipeline(question=question, context=context)["answer"]
    elif mode == "QA":
        answer = "I'm unable to find relevant information from the provided documents. Please provide more context or rephrase your question."
    else:
        if relevant_docs:
            context = " ".join(relevant_docs)
            prompt = (
                f"Given the context below, provide a detailed and accurate response to the question.\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            prompt = f"Provide a detailed and accurate response to the following question.\n\nQuestion: {question}\n\nAnswer:"
        
        response = generative_pipeline(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,  # Adjust temperature for more focused answers
            top_p=0.9,       # Adjust top-p for more coherent answers
            pad_token_id=50256,
            truncation=True
        )[0]["generated_text"]
        
        # Filter incomplete or irrelevant responses
        answer = response.split("Answer:")[-1].strip()
        if not answer or len(answer) < 10:
            answer = "I'm unable to generate a relevant answer at the moment. Please try asking in a different way."

    return answer

# Function to log chat history
def log_chat(question, answer):
    chat_history = {"questions": st.session_state['questions'], "answers": st.session_state['answers']}
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f)
