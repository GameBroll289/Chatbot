import pandas as pd

doc = pd.read_excel('/home/ahmed/Desktop/Chatbot/Pharmacy Inventory Data Generation.xlsx')

drop_columns = ['ID']

doc = doc.drop(columns=drop_columns)

print(doc)