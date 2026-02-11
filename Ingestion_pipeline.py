import pandas as pd
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

Make_DataBase = True

if (Make_DataBase == True):

    # 1. Load the data (Do NOT drop the ID column this time!)
    doc = pd.read_excel('C:/Users/Ahmed/Desktop/Chatbot/Pharmacy Inventory Data Generation DES.xlsx')

    langchain_chunks = []

    # 2. Iterate
    for index, row in doc.iterrows():
        
        # 3. SELECTIVE CHUNKING
        # We purposefully DO NOT include {row['ID']} in this string.
        # This effectively "drops" it from the text the AI reads.
        text_content = (
            f"Medicine: {row['Medicine Name']}. "
            f"Description: {row['Description (Symptoms & Uses)']}. "
            f"Price: ${row['Price ($)']}. "
            f"Quantity: {row['Stock Quantity']}. "
            f"Expiry Date: {row['Expiry Date']}."
        )
        
        # 4. METADATA
        # We CAN use {row['ID']} here because we didn't delete it from the dataframe.
        # This is great for your app to find the original row later.
        new_chunk = Document(
            page_content=text_content,
            metadata={
                "source": "Pharmacy_Data", 
                "row_id": row['ID']  # This works now!
            }
        )
        
        langchain_chunks.append(new_chunk)
        
    # 1. Initialize the Local Embedding Model
    # This downloads the model the first time you run it (stored locally thereafter).
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Create the Vector Store
    # This does the heavy lifting: 
    #   a. Takes your text chunks
    #   b. Converts them to numbers (embeddings) using the model
    #   c. Saves them into a local folder called "./chroma_db"
    vector_store = Chroma.from_documents(
        documents=langchain_chunks, 
        embedding=embedding_model,
        persist_directory="./chroma_db"  # Saves to your hard drive
    )

    print("Success! Data embedded and saved locally.")

else:
    # 1. Define the SAME embedding model used during creation
    # (It needs this to convert new user queries into the same number format)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load the existing Vector Store
    # Notice we just point to the directory. We do NOT pass 'documents' here.
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )

# Ask a question based on your Excel file
#query = "What is the price of Aspirin?"
query = "What medicines are available for headache?"

# Retrieve the top 3 most similar chunks
results = vector_store.similarity_search(query, k=3)

print("\n--- Search Results ---")
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Source Row ID: {doc.metadata['row_id']}")
    print("---")