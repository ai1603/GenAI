import os
import pypdf
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader 
from langchain_community.document_loaders import UnstructuredExcelLoader
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# client = OpenAI()
llm = ChatOpenAI(model="gpt-4o", temperature=0,  max_tokens=200, api_key= OPENAI_API_KEY)

def load_documents_from_directory(file_path):
    """"Load docuements from various file formats"""
    
    # file_path_str = Path(file_path)
    file_path_str = str(file_path)
    
    if file_path_str.endswith(".pdf"):
        try:
            loader = PyPDFLoader(file_path_str)
            document = loader.load()
            return document
        except Exception as e:
            print(f"Error loading document as {e}")
            return None
    
    elif file_path_str.endswith((".doc", ".docx")):
        try:
            loader = Docx2txtLoader(file_path_str)
            document = loader.load()
            return document
        except Exception as e:
            print(f"Error loading document as {e}")
            return None
        
    elif file_path_str.endswith((".xlsx", "xls")):
        try:
            loader = UnstructuredExcelLoader(file_path_str)
            document = loader.load()
            return document
        except Exception as e:
            print(f"Error loading document as {e} ")
            return None
    elif file_path_str.endswith(".txt"):
        try:
            loader = TextLoader(file_path_str)
            document = loader.load()
            return document
        except Exception as e:
            print(f"Error loading the document as {e}")
            return None
    else:
        print(f"Un supported file format. Supported file formats: .pdf, .dcox, .doc, .xls, .xlsx")
        return None

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_faiss_index(documents):
    """Create FAISS vector store for documents"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        print("Failed to create the index")
        return None

def retrieve_doc(vec_store):
    retriever = vec_store.as_retriever(search_type = "similarity",
                                       search_kwargs={"k": 4})
    return retriever

def create_qa_chain(llm, retriever):
    """Create a question-answer chain"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, say so.\n\nContext: {context} "),
        ("human", "{input}")
    ])
    #create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    #create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def validate_file_format(file_path):
    """Validate if the file format is supported. """
    supported_documents = [".pdf", ".docx", ".doc", ".xls", ".xlsx", ".txt"]
    file_extension = file_path.suffix.lower()
    return file_extension in supported_documents
    
if __name__ == "__main__":
    
    #Get file path from the user
    file_path_raw = input("Please enter file path: ")
    file_path = Path(file_path_raw)
    
    #file path validation whether pdf, word, excel or website or 
    print("File path:", file_path)
    
    if not file_path.exists():
        print(f"file {file_path} doesn't exist and please place the file")
        exit(1)
    
    #Validate the file formate 
    if not validate_file_format(file_path):
        print(f"Error: Unsupported file format. Supported formats: .pdf, .doc, .docx, .xlsx, .xls, .txt")
        exit(1)
        
    #Load documents
    documents = load_documents_from_directory(file_path)
    if documents is None:
        print("Failed to load documents. Exiting.")
        exit(1)
    print(f"Loaded {len(documents)} document pages")
    
    #Split the documents into chunks
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    #Create vector store
    vec_store = create_faiss_index(chunks)
    if vec_store is None:
        print("Failed to create vector store. Exiting.")
        exit(1)
    
    #Create retriever 
    retriever = retrieve_doc(vec_store)
    
    #Create QA chain
    qa_chain_instance = create_qa_chain(llm, retriever)
    
    print("\nDocument processed successfully! You can now ask questions.")
    print("Type 'exit' to quit the program.")
    
    while True:
        query = input("\nAsk question: ")
        if query.lower() == "exit":
            break
        try:
            response = qa_chain_instance.invoke({"input": query})
            # respons = qa_chain.invoke(query)
            print(f"Answer: {response['answer']}")
        except Exception as e:
            print(f"Error processing query: {e}")    
    