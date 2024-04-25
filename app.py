from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import  SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

st.title("Q&A ChatBot: Using RAG System ")

prompt = st.text_area("Ask any questions from 'Leave No Context Behind Paper' here:...")
# Load API key from file
with open('keys/.gemini_API_key.txt') as f:
    API_KEY = f.read().strip()

# Correct usage of API key and model name
chat_model = ChatGoogleGenerativeAI(google_api_key=API_KEY, 
                                   model="gemini-1.5-pro-latest")

# Initialize output parser
output_parser = StrOutputParser()

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, 
                                               model="models/embedding-001")

# Initialize ChromaDB connection
db_connection = Chroma(persist_directory="./chromadb", embedding_function=embedding_model)

# Initialize retriever
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Initialize chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from the user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

# Define document formatting function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


# Display a button to ask the question
if st.button("Submit"):
    # Assuming rag_chain.invoke() returns the answer to the user's question
    response =rag_chain.invoke(prompt)  # Replace this with actual code to invoke rag_chain
    st.markdown(response)