import os
import uuid
import boto3
import streamlit as st

# AWS S3 Configuration
s3_client = boto3.client("s3")
BUCKET_NAME = "yojitha-chat-with-pdf"

# Ensure AWS Region is Set
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Load AWS credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

# Initialize Bedrock Client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# LangChain Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

folder_path = "/tmp/"

# Function to clean file names (removes spaces & special characters)
def clean_file_name(file_name):
    return "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in file_name)

# Split text into chunks
def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Fetch FAISS indexes from S3 and download if needed
def list_faiss_indexes():
    """Fetch and list FAISS indexes from S3, ensuring they are downloaded locally."""
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    
    if "Contents" in response:
        indexes = sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"] if obj["Key"].endswith(".faiss")))
        
        # Ensure each index is downloaded locally
        for index in indexes:
            local_faiss_path = os.path.join(folder_path, f"{index}.faiss")
            local_pkl_path = os.path.join(folder_path, f"{index}.pkl")
            
            if not os.path.exists(local_faiss_path):
                s3_client.download_file(BUCKET_NAME, f"faiss_files/{index}.faiss", local_faiss_path)
            
            if not os.path.exists(local_pkl_path):
                s3_client.download_file(BUCKET_NAME, f"faiss_files/{index}.pkl", local_pkl_path)
        
        return indexes

    return []

# Create or update FAISS vector store only when a new file is uploaded
def create_vector_store(file_name, documents):
    local_folder = "/tmp"
    faiss_folder = os.path.join(local_folder, file_name)
    os.makedirs(faiss_folder, exist_ok=True)

    faiss_index_path = os.path.join(faiss_folder, "index")
    pkl_path = os.path.join(faiss_folder, "index.pkl")

    # Avoid reprocessing existing FAISS indexes
    if file_name in list_faiss_indexes():
        return

    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    vectorstore_faiss.save_local(index_name="index", folder_path=faiss_folder)

    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

# Initialize the LLM
def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

# Retrieve answers using FAISS (Fixed to Prevent Infinite Loop)
def get_response(llm, vectorstore, question):
    """Retrieve answers using FAISS without infinite loops and ensure document retrieval."""
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say you don't know.
    
    <context>
    {context}
    </context>

    Question: {question}
    
    Assistant:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # ✅ FIX: Ensure documents are retrieved before answering
    response = qa.invoke({"query": question})
    
    # Debugging: Print retrieved documents
    retrieved_docs = response.get("source_documents", [])
    st.write(f"🔍 Retrieved {len(retrieved_docs)} document(s) from FAISS for question: '{question}'")
    
    if not retrieved_docs:
        return "I couldn't find relevant information in the document."

    return response["result"]

# Main Streamlit App
def main():
    st.title("Chat with Your PDF")

    # File Upload Section (Only Process Once)
    st.subheader("Upload PDF to Add to the Knowledge Base")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_file_name = os.path.splitext(uploaded_file.name)[0]
            clean_name = clean_file_name(original_file_name)

            saved_file_name = os.path.join("/tmp", f"{clean_name}.pdf")
            with open(saved_file_name, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()

            splitted_docs = split_text(pages)

            # Process & create FAISS index only ONCE
            if clean_name not in list_faiss_indexes():
                create_vector_store(clean_name, splitted_docs)
                st.success(f"Successfully processed {uploaded_file.name}!")

    # Question Answering Section (Separate from Uploading)
    st.subheader("Ask Questions from the Knowledge Base")

    faiss_indexes = list_faiss_indexes()

    if not faiss_indexes:
        st.error("No FAISS indexes found in S3. Please upload PDFs first.")
        return

    selected_index = st.selectbox("Select a FAISS index", faiss_indexes)

    # Dynamic Placeholder with Document Name
    question_placeholder = f"Ask a question about {selected_index}" if selected_index else "Ask a question"
    question = st.text_input(question_placeholder)

    if st.button("Ask Question"):
        with st.spinner("Finding the best answer..."):
            faiss_index = FAISS.load_local(
                index_name=selected_index,
                folder_path=folder_path,
                embeddings=bedrock_embeddings,
                allow_dangerous_deserialization=True
            )

            answer = get_response(get_llm(), faiss_index, question)

        st.success("Here's the answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
