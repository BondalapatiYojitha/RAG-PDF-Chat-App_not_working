import os
import uuid
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS S3 Configuration
s3_client = boto3.client("s3")
BUCKET_NAME = "yojitha-chat-with-pdf"

# Ensure AWS Region is Set
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Initialize Bedrock Client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Initialize Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

folder_path = "/tmp/"

# Utility: Generate Unique ID for Uploaded Files
def get_unique_id():
    return str(uuid.uuid4())

# Utility: Clean File Name
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

# Ensure FAISS index is downloaded
def ensure_faiss_downloaded(index_name):
    """Ensures that the FAISS index files are downloaded before being used."""
    local_faiss_path = os.path.join(folder_path, f"{index_name}.faiss")
    local_pkl_path = os.path.join(folder_path, f"{index_name}.pkl")

    # Download if missing
    if not os.path.exists(local_faiss_path):
        try:
            s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.faiss", local_faiss_path)
        except:
            st.error(f"FAISS index file `{index_name}.faiss` not found in S3.")
            return False

    if not os.path.exists(local_pkl_path):
        try:
            s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.pkl", local_pkl_path)
        except:
            st.warning(f"Metadata file `{index_name}.pkl` is missing but proceeding without it.")

    return True

# Create FAISS Index
def create_vector_store(file_name, documents):
    local_folder = "/tmp"
    faiss_folder = os.path.join(local_folder, file_name)
    os.makedirs(faiss_folder, exist_ok=True)

    faiss_index_path = os.path.join(faiss_folder, "index")
    pkl_path = os.path.join(faiss_folder, "index.pkl")

    if vector_store_exists(file_name):
        ensure_faiss_downloaded(file_name)

        existing_vectorstore = FAISS.load_local(
            index_name="index",
            folder_path=faiss_folder,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True  # ✅ FIXED: Safe deserialization
        )

        new_vectorstore = FAISS.from_documents(documents, bedrock_embeddings)
        existing_vectorstore.merge_from(new_vectorstore)
        existing_vectorstore.save_local(index_name="index", folder_path=faiss_folder)
    else:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        vectorstore_faiss.save_local(index_name="index", folder_path=faiss_folder)

    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

    return True

# List FAISS indexes from S3
def list_faiss_indexes():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    if "Contents" in response:
        return sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"] if obj["Key"].endswith(".faiss")))
    return []

# Load FAISS index from local storage
def load_faiss_index(index_name):
    """Load FAISS index after ensuring it is downloaded."""
    if ensure_faiss_downloaded(index_name):
        return FAISS.load_local(
            index_name=index_name,
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    return None

# Merge all FAISS indexes into a single vectorstore
def merge_all_indexes(excluded_index=None):
    all_faiss_indexes = list_faiss_indexes()
    combined_vectorstore = None

    for index in all_faiss_indexes:
        if index == excluded_index:
            continue  

        faiss_index = load_faiss_index(index)
        if faiss_index:
            if combined_vectorstore is None:
                combined_vectorstore = faiss_index
            else:
                combined_vectorstore.merge_from(faiss_index)

    return combined_vectorstore

# Initialize LLM
def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

# Perform retrieval and generate response
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

    response = qa.invoke({"query": question})
    retrieved_docs = response.get("source_documents", [])
    
    return response, retrieved_docs

# Streamlit App
def main():
    st.title("Chat with Your PDF")

    # Question Answering
    st.subheader("Ask Questions from the Knowledge Base")

    faiss_indexes = list_faiss_indexes()
    if not faiss_indexes:
        st.error("No FAISS indexes found. Please upload PDFs first.")
        return

    selected_index = st.selectbox("Select a FAISS index", faiss_indexes)
    question = st.text_input(f"Ask a question about {selected_index}")

    if st.button("Ask Question"):
        with st.spinner("Finding the best answer..."):
            selected_vectorstore = load_faiss_index(selected_index)
            if selected_vectorstore:
                response, retrieved_docs = get_response(get_llm(), selected_vectorstore, question)

                if retrieved_docs:
                    st.success("Here's the answer:")
                    st.write(response["result"])
                    return

                st.error(f"❌ Couldn't find relevant information in {selected_index} or any other document.")
            else:
                st.error(f"❌ FAISS index `{selected_index}` could not be loaded. Please check if it exists in S3.")

if __name__ == "__main__":
    main()
