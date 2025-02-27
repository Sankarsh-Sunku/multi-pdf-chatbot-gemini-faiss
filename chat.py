import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
# login(token="{yourToken}") # If you are using a model from Huggingface only
# repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Name of the model which you are working

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text +=page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store_for_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(chunks, 
                                    embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    context:\n {context}?\n
    question: \n{question}\n

    Answer:
    """

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
        prompt |
        llm |
        StrOutputParser()
    )

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain()
    docs = new_db.similarity_search(user_question, k =2)
    response = chain.invoke(
        {"context":docs, "question": user_question})
    print(response)
    return response


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
    st.title("Multi PDF ChatAgent")

    response = None
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝", )
    # if user_question:
    #     response = user_input(user_question)
    
    if user_question.strip():  # Ensure input is not empty
        response = user_input(user_question)  # Process only when button is clicked
    else:
        st.warning("Please enter a question before submitting.")

    with st.sidebar:
        st.title("📁 PDF File's Section")
        uploaded_file = st.file_uploader("Upload a PDF before asking a question", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                text = get_pdf_text(uploaded_file)
                chunks = get_chunks(text)
                get_vector_store_for_chunks(chunks)
                st.success("PDF Text Extraction and Vector Indexing Completed Successfully")
        st.write("----")
    
    st.header("Answer : ")
    st.subheader(response)

    
if __name__ == "__main__":
    main()
