import streamlit as st
import os
# from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
import keyring
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import AzureOpenAI

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["AZURE_OPENAI_API_KEY"] = "a121608bfa654d7bbb4ff9718ecba306"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"

def get_conversational_chain():
    prompt_template = """
    1. Your Job is to convert a user question to SQL Query. You have to give the query so that it can be used on Microsoft SQL server SSMS.
    2. There is only one table with table name Sentiment_Data, It has 8 columns: They are
        DeviceFamilyName - which is the name of the device | Sentence - the review for that device | keyword - the main keyword extracted from the review | Aspect - The aspect that consumer talks about in the review, it may be performance, battery and so on | Sentiment - It may be positive or negative or neutral based on the sentence | Sentiment_Score - It will be 1, 0 or -1 based on the sentiment | Review_Count - It will be 1 for each sentence or each row
    3. Sentiment mark of a device is calculated by number of positive reviews subtracted by number of negative reviews of that device
    4. Net sentiment of a device is sentiment mark of a device divided by total reviews of that device. It should be in percentage
           Example To calculate net sentiment of a device : 
            SELECT DeviceFamilyName, (SUM(Sentiment_Score) / SUM(Review_Count)) * 100 AS Net_Sentiment
            FROM Sentiment_Data
            WHERE DeviceFamilyName = 'Mentioned Device'
    5. The aspect-wise sentiment of a device is determined by dividing the sentiment mark of a particular aspect for that device by the total number of reviews regarding that aspect on the device. This calculation encompasses all aspects listed in the aspect column, not just performance or battery life.  It should be in percentage. Sort it by Review Count
            Example To calculate Aspect wise sentiment of a device : 
                SELECT Aspect, (SUM(Sentiment_Score) / SUM(Review_Count)) * 100 AS Aspect_Sentiment
                FROM Sentiment_Data
                WHERE DeviceFamilyName = 'Mentioned device'
                GROUP BY Aspect
                ORDER BY SUM(Review_Count)
    6. Top device is based on Sentiment_Score i.e., the device which have highest sum(Sentiment_Score)
    7. Whenever user ask about top devices and their aspect wise sentiment. Give them top device and then give them the aspect wise sentiment of that device. Make sure you use subqueries
    8. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = AzureChatOpenAI(
    azure_deployment="Verbatim-Synthesis",
    api_version='2023-12-01-preview')
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def query(user_question, vector_store_path="faiss_index_new"):
    # Initialize the embeddings model
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
    
    # Load the vector store with the embeddings model
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
    # Rest of the function remains unchanged
    chain = get_conversational_chain()
    docs = vector_store.similarity_search(user_question,max_results = None)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response["output_text"])
    return response["output_text"]

def initialize_credentials():
    server = 'qz7skm1a4o.database.windows.net'
    database = 'ReviewSentimentAnalysis'
    tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47'
    credential = InteractiveBrowserCredential(tenant_id=tenant_id)
    driver = '{ODBC Driver 17 for SQL Server}'
    connection_string = 'DRIVER=' + driver + ';Server=' + server + ';Database=' + database + ';Authentication=ActiveDirectoryInteractive'
    params = urllib.parse.quote_plus(connection_string)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    return connection_string, params, engine

def main():
    st.title("Data Pull")
    user_question = st.text_input("Ask a question")
    if user_question:
        SQL_Query = query(user_question)
        initialize_credentials()
        connection_string, params, engine = initialize_credentials()
        conn = pyodbc.connect(connection_string, authentication=InteractiveBrowserCredential(tenant_id='72f988bf-86f1-41af-91ab-2d7cd011db47'))
        data = pd.read_sql(SQL_Query, conn)
        st.subheader("Data:")
        st.write(data)
        llm_1 = model = AzureChatOpenAI(
        azure_deployment="Verbatim-Synthesis",
        api_version='2023-12-01-preview')
        sdf_1 = SmartDataframe(data, config={"llm": llm_1})
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        sdf_1.chat("Understand the data, column names and plot suitable graph")
        plt.rcParams.update({'font.size': 12})
        fig = plt.gcf()  # Get current figure
        plt.close()  # Close the plot to avoid displaying it directly
        st.subheader("Plot:")
        st.pyplot(fig)
        
        


if __name__ == "__main__":
    main()