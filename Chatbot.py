import pandas as pd
from langchain_core.documents import Document

# 1. Load the data (Do NOT drop the ID column this time!)
doc = pd.read_excel('/home/ahmed/Desktop/Chatbot/Pharmacy Inventory Data Generation.xlsx')

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

print(f"Created {len(langchain_chunks)} chunks.")
print("Example Chunk Content:", langchain_chunks[0].page_content)
print("Example Chunk Metadata:", langchain_chunks[0].metadata)