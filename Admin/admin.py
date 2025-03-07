import os
import uuid
import boto3
import streamlit as st

# AWS S3 Configuration
s3_client = boto3.client("s3")
#BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_NAME = "yojitha-chat-with-pdf"
# Explicitly Set AWS Region
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Load AWS credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

st.write(f"aws_access_key  : {aws_access_key if aws_access_key else 'MISSING'}")
st.write(f"aws_secret_key  : {aws_secret_key if aws_secret_key else 'MISSING'}")
st.write(f"aws_region  : {aws_region if aws_region else 'MISSING'}")


# Initialize Bedrock Client with credentials
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Import Bedrock Embeddings from `langchain-aws`
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Initialize Bedrock Embeddings (✅ FIXED)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

## main method
def main():
    st.write("This is Admin Site for Chat with PDF  : ")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        st.write("===================")
        st.write(splitted_docs[0])
        st.write("===================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully")
        else:
            st.write("Error!! Please check logs.")



if __name__ == "__main__":
    main()
