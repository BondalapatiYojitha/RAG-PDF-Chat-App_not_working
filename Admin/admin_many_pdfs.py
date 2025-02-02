import os
import uuid
import boto3
import streamlit as st
import shutil

# AWS S3 Configuration
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Explicitly Set AWS Region
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Initialize Bedrock Client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")  # Ensure region is always set
)

# LangChain Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Initialize Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client,
    model_kwargs={"input_type": "search_document"}
)

# Generate Unique ID
def get_unique_id():
    return str(uuid.uuid4())

# Split text into chunks
def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

# Function to check if a vector store already exists in S3
def vector_store_exists(request_id):
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=f"faiss_files/{request_id}.faiss")
        return True
    except:
        return False

# Merge an existing FAISS index with new vectors
def merge_vector_stores(existing_faiss, new_faiss):
    existing_faiss.merge_from(new_faiss)

# Create or Update FAISS Vector Store
def create_vector_store(request_id, documents):
    local_folder = "/tmp"
    faiss_path = os.path.join(local_folder, f"{request_id}.faiss")
    pkl_path = os.path.join(local_folder, f"{request_id}.pkl")

    # Check if existing vector store is in S3
    if vector_store_exists(request_id):
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{request_id}.faiss", faiss_path)
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{request_id}.pkl", pkl_path)

        # Load existing FAISS
        existing_vectorstore = FAISS.load_local(index_name=faiss_path, embeddings=bedrock_embeddings)

        # Create new FAISS from documents
        new_vectorstore = FAISS.from_documents(documents, bedrock_embeddings)

        # Merge with existing vectorstore
        merge_vector_stores(existing_vectorstore, new_vectorstore)
        existing_vectorstore.save_local(index_name=faiss_path, folder_path=local_folder)

    else:
        # Create new FAISS if none exists
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        vectorstore_faiss.save_local(index_name=faiss_path, folder_path=local_folder)

    # Upload to S3
    s3_client.upload_file(faiss_path, BUCKET_NAME, f"faiss_files/{request_id}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{request_id}.pkl")

    return True

# Streamlit UI
def main():
    st.title("Admin Panel for Chat with PDF")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            request_id = get_unique_id()
            st.write(f"Processing PDF: {uploaded_file.name}")
            st.write(f"Request ID: {request_id}")

            saved_file_name = os.path.join("/tmp", f"{request_id}.pdf")
            with open(saved_file_name, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                loader = PyPDFLoader(saved_file_name)
                pages = loader.load_and_split()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue

            st.write(f"Total Pages: {len(pages)}")

            # Split Text
            splitted_docs = split_text(pages)
            st.write(f"Splitted Docs: {len(splitted_docs)}")

            # Process & Store Vectors
            st.write("Creating the Vector Store...")
            result = create_vector_store(request_id, splitted_docs)

            if result:
                st.success(f"Successfully processed {uploaded_file.name}!")
            else:
                st.error(f"Error processing {uploaded_file.name}!")

if __name__ == "__main__":
    main()
