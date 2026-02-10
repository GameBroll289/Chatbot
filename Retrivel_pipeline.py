import os
# This tells Hugging Face to use local files only and NEVER ping the internet
os.environ['HF_HUB_OFFLINE'] = '1'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from flask import Flask, request, jsonify

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the existing Vector Store
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model
)

from dotenv import load_dotenv # Import the new tool

# 1. Load the secret .env file
load_dotenv()

# Initialize the LLM (Llama 3 is great for reasoning)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# Define a prompt template to structure the input for the LLM
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a chain that combines the retrieved documents into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retriever from your vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create the final retrieval chain that connects the retriever and the document chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

while True:
    query = input("Enter your search query (or 'q' to exit): ")
    if query.lower() == 'q':
        break
    else:
        # The input to the new chain is a dictionary
        response = retrieval_chain.invoke({"input": query})

        print("\n--- AI Answer ---")
        # The answer is now in the 'answer' key of the response dictionary
        print(response['answer'])

#HI! This is the Retrieval Pipeline. It connects to the vector store we created in the Ingestion Pipeline, retrieves relevant chunks based on a user query, and then uses an LLM to generate an answer based on those chunks.