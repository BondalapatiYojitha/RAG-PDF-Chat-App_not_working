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

# Check if FAISS index exists in S3
def vector_store_exists(file_name):
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=f"faiss_files/{file_name}.faiss")
        return True
    except:
        return False

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

        # FIX: Load FAISS safely with deserialization enabled
        existing_vectorstore = FAISS.load_local(
            index_name="index",
            folder_path=faiss_folder,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )

        new_vectorstore = FAISS.from_documents(documents, bedrock_embeddings)
        existing_vectorstore.merge_from(new_vectorstore)

        existing_vectorstore.save_local(index_name="index", folder_path=faiss_folder)

    else:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        vectorstore_faiss.save_local(index_name="index", folder_path=faiss_folder)

    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

# Function to list FAISS indexes in S3
def list_faiss_indexes():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    if "Contents" in response:
        return sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"]))
    return []

# Function to download a selected FAISS index from S3
def load_selected_index(selected_index):
    s3_client.download_file(BUCKET_NAME, f"faiss_files/{selected_index}.faiss", f"{folder_path}{selected_index}.faiss")
    s3_client.download_file(BUCKET_NAME, f"faiss_files/{selected_index}.pkl", f"{folder_path}{selected_index}.pkl")

# Initialize the LLM
def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

# Retrieve answers using FAISS
def get_response(llm, vectorstore, question):
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

    answer = qa({"query": question})
    return answer['result']

# Main Streamlit App
def main():
    st.title("Chat with Your PDF")

    # File Upload Section
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
            create_vector_store(clean_name, splitted_docs)
            st.success(f"Successfully processed {uploaded_file.name}!")

    # Question Answering Section
    st.subheader("Ask Questions from the Knowledge Base")

    faiss_indexes = list_faiss_indexes()
    selected_index = st.selectbox("Select a FAISS index", faiss_indexes)

    faiss_index_path = os.path.join(folder_path, selected_index, "index.faiss")

    if not os.path.exists(faiss_index_path):
        st.error(f"FAISS index file not found: {faiss_index_path}. Please upload the document first.")
        return

    # Load FAISS safely
    faiss_index = FAISS.load_local(selected_index, folder_path, bedrock_embeddings, allow_dangerous_deserialization=True)
    question = st.text_input(f"Ask a question about {selected_index}")

    if st.button("Ask Question"):
        answer = get_response(get_llm(), faiss_index, question)
        st.write(answer)

if __name__ == "__main__":
    main()
