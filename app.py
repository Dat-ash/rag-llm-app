import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from info import show_info, categories
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import re
import os
import sys


def main():
    print("Initializing application...")
    load_dotenv()
    st.set_page_config(page_title="Chat with Legal Documents 📜🏛️", 
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    st.header("Chat with Legal Documents 📜🏛️:books:")
    st.markdown("Upload, ask, and get insights from your legal PDFs")

    # Show Tip with background color 
    st.markdown("""
<div style="background-color: #d1ecf1; border-left: 5px solid #0c5460; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <strong>💡 Tip:</strong> Choosing the right category helps the AI give you more accurate answers!
</div>
""", unsafe_allow_html=True)
    # st.markdown("💡 Tip: Choosing the right category helps the AI give you more accurate answers!")

    # Show categories in app
    # Show info icon
    show_info()
    st.selectbox("Choose a document category", list(categories.keys()))

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        print("user input initialized|||||")
        handle_userinput(user_question)
        print("user input completed|||||")


    with st.sidebar:
        st.subheader("📂 Legal Document Upload")
        pdf_docs = st.file_uploader("Upload your legal PDFs (e.g., contracts, claims, agreements) and click 'Process' to begin", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                print("vectorstore initialized||||||")
                vectorstore = get_vectorstore(text_chunks)
                print(f"Vector store created with {vectorstore.index.ntotal} embeddings")

                # create conversation chain
                print("conversation initialized||||||")
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.chat_history = ChatMessageHistory()
                print("conversation completed=====")
                st.success("Documents processed successfully!")

def get_pdf_text(pdf_docs):   #final

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):   #final
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks")
    return chunks

def get_vectorstore(text_chunks):   #final
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model="text-embedding-3-large",
        openai_api_version = "2024-02-01"
    )

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("Vector store created successfully")
    return vectorstore

def get_conversation_chain(vectorstore):  #final
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-05-13",
        azure_deployment = "gpt-4o",
        openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.3,
        streaming=True
    )
    print("LLM initialized successfully")
    
    # 1. Create history-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the conversation, formulate a search query to look up")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm,
        vectorstore.as_retriever(),
        contextualize_q_prompt
    )
    
    # 2. Create document chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question based only on the following context:
        {context}
        
        If you don't know the answer, just say you don't know."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Combine into final chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def handle_userinput(user_question):   #final
    if 'conversation' not in st.session_state:
        st.error("Please process documents first")
        return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()
    
    try:
        # Invoke the chain
        response = st.session_state.conversation.invoke({
            "input": user_question,
            "chat_history": st.session_state.chat_history.messages
        })
        
        # Update chat history
        st.session_state.chat_history.add_user_message(user_question)
        print("user_question-->", user_question)
        print("\n=================\n")

        print("response-->", response)
        print("\n=================\n")
        print("response[answer]-->", response["answer"])

        # # This code gives whole "System: ... Human: ... Assistant: ..." block
        # st.session_state.chat_history.add_ai_message(response["answer"])

        # This code give only assistant block
        assistant_response = response["answer"]
        if "Assistant:" in assistant_response:
            assistant_response = assistant_response.split("Assistant:")[-1].strip()

        # Optional: strip System & Human parts if present
        assistant_response = re.sub(r"(System|Human):.*?(?=Assistant:)", "", response["answer"], flags=re.DOTALL).split("Assistant:")[-1].strip()

        st.session_state.chat_history.add_ai_message(assistant_response)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()