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
import requests

print("Loading AI...")
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

print("AI Ready!")
# --- [META CONFIGURATION] ---
app = Flask(__name__)

# PASTE YOUR KEYS FROM PHASE 1 HERE
WHATSAPP_TOKEN = "EAAMgZATsP0dsBQo5MhcUffTh5D1L6ZAdigS1A7PiJwwtmZAbij7iQznG1pzZABFDFzuZATYaOJ5UaJCDjlZBMqHSrNhkZAslmoTOESBa25bxhfLyrW2nqHOi4QGthm0AvpuBIwm3kCEvRIjgeR97GpofGrQcIXM2GZAIWfIOIfKztVq8JMCKOprjM5gmvncGdaKDSgFfZBCnXZBqX06S9XCqrQKeC0zSXYYJwWJ3cUqGmvTEI0giZBBzwv6Eaaxg5raZCwBPMYPncVRbZBDUr5u21QnjI" 
PHONE_NUMBER_ID = "963898066811396"
VERIFY_TOKEN = "Pak" # You can keep this as 'blue_zebra' or change it

def send_whatsapp_message(recipient_id, text):
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    data = {"messaging_product": "whatsapp", "to": recipient_id, "text": {"body": text}}
    
    # 1. Capture the response
    response = requests.post(url, headers=headers, json=data)
    
    # 2. Print the diagnosis
    print(f"\n--- META REPLY LOG ---")
    print(f"Status Code: {response.status_code}") # 200 = Good, 400/401 = Bad
    print(f"Error Details: {response.text}")
    print("----------------------\n")

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    # Meta sends a GET request to verify this URL is yours
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200 # We echo the challenge back
    return "Forbidden", 403

@app.route('/webhook', methods=['POST'])
def receive_message():
    data = request.get_json()
    try:
        if data.get("entry"):
            changes = data["entry"][0]["changes"][0]["value"]
            if "messages" in changes:
                msg_info = changes["messages"][0]
                sender_id = msg_info["from"]
                user_text = msg_info["text"]["body"]
                
                print(f"User: {user_text}")
                
                # Ask RAG
                response = retrieval_chain.invoke({"input": user_text})
                ai_answer = response['answer']
                
                # Reply
                send_whatsapp_message(sender_id, ai_answer)
    except Exception as e:
        print(f"Error: {e}")
    return "Processed", 200

if __name__ == '__main__':
    app.run(port=5000)