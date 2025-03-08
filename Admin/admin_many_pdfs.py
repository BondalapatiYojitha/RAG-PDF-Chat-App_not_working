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

# Validate AWS Credentials
if not aws_access_key or not aws_secret_key:
    st.error("AWS credentials are missing! Please configure them in your environment.")

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

# Initialize Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

# Function to clean file names (removes spaces & special characters)
def clean_file_name(file_name):
    return "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in file_name)

# Check if FAISS index exists in S3
def vector_store_exists(file_name):
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=f"faiss_files/{file_name}.faiss")
        return True
    except:
        return False

# Merge FAISS vector stores correctly
def merge_vector_stores(existing_faiss, new_faiss):
    texts = [new_faiss.docstore[i] for i in new_faiss.index_to_docstore_id.keys()]
    vectors = list(new_faiss.index_to_docstore_id.keys())
    existing_faiss.add_texts(texts, vectors)

# Create or Update FAISS Vector Store
def create_vector_store(file_name, documents):
    local_folder = "/tmp"
    faiss_folder = os.path.join(local_folder, file_name)
    os.makedirs(faiss_folder, exist_ok=True)

    faiss_index_path = os.path.join(faiss_folder, "index")
    pkl_path = os.path.join(faiss_folder, "index.pkl")

    if vector_store_exists(file_name):
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{file_name}.faiss", faiss_index_path + ".faiss")
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{file_name}.pkl", pkl_path)

        existing_vectorstore = FAISS.load_local(index_name="index", folder_path=faiss_folder, embeddings=bedrock_embeddings)
        new_vectorstore = FAISS.from_documents(documents, bedrock_embeddings)

        merge_vector_stores(existing_vectorstore, new_vectorstore)

        existing_vectorstore.save_local(index_name="index", folder_path=faiss_folder)

    else:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        vectorstore_faiss.save_local(index_name="index", folder_path=faiss_folder)

    if not os.path.exists(faiss_index_path + ".faiss"):
        raise FileNotFoundError(f"FAISS index file not found: {faiss_index_path}.faiss")

    if not os.path.exists(pkl_path):
        with open(pkl_path, "wb") as f:
            pass  

    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

    return True

# Streamlit UI
def main():
    st.title("Admin Panel for Chat with PDF")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_file_name = os.path.splitext(uploaded_file.name)[0]  # Get file name without extension
            clean_name = clean_file_name(original_file_name)  # Clean file name
            
            st.write(f"Processing PDF: {uploaded_file.name}")
            st.write(f"File Identifier: {clean_name}")

            saved_file_name = os.path.join("/tmp", f"{clean_name}.pdf")
            with open(saved_file_name, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                loader = PyPDFLoader(saved_file_name)
                pages = loader.load_and_split()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue

            st.write(f"Total Pages: {len(pages)}")

            splitted_docs = split_text(pages)
            st.write(f"Splitted Docs: {len(splitted_docs)}")

            st.write("Creating the Vector Store...")
            try:
                result = create_vector_store(clean_name, splitted_docs)
                if result:
                    st.success(f"Successfully processed {uploaded_file.name}!")
                else:
                    st.error(f"Error processing {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error during vector store creation: {e}")

if __name__ == "__main__":
    main()
