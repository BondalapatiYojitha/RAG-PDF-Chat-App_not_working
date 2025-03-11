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

# Utility: Clean File Name
def clean_file_name(file_name):
    return "".join(c if c.isalnum() or c in ('.', '_') else "_" for c in file_name)

# Split text into chunks
def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Check if FAISS index exists in S3
def faiss_exists_in_s3(index_name):
    """Check if FAISS index file exists in S3 before attempting to download."""
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=f"faiss_files/{index_name}.faiss")
        return True
    except:
        return False

# Ensure FAISS index is downloaded
def ensure_faiss_downloaded(index_name):
    """Ensure that the FAISS index files are downloaded before being used."""
    local_faiss_path = os.path.join(folder_path, f"{index_name}.faiss")
    local_pkl_path = os.path.join(folder_path, f"{index_name}.pkl")

    if not faiss_exists_in_s3(index_name):
        st.error(f"⚠️ FAISS index `{index_name}` not found in S3. Please upload the document again.")
        return False

    # Download if missing
    if not os.path.exists(local_faiss_path):
        s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.faiss", local_faiss_path)

    if not os.path.exists(local_pkl_path):
        try:
            s3_client.download_file(BUCKET_NAME, f"faiss_files/{index_name}.pkl", local_pkl_path)
        except:
            st.warning(f"⚠️ Metadata file `{index_name}.pkl` is missing. Proceeding without it.")

    return True

# Create FAISS Index (Only Runs During PDF Upload)
def create_vector_store(file_name, documents):
    """Creates and uploads FAISS index. Runs **ONLY ONCE** during PDF upload."""
    local_folder = "/tmp"
    faiss_folder = os.path.join(local_folder, file_name)
    os.makedirs(faiss_folder, exist_ok=True)

    faiss_index_path = os.path.join(faiss_folder, "index")
    pkl_path = os.path.join(faiss_folder, "index.pkl")

    # Skip reprocessing if FAISS already exists
    if faiss_exists_in_s3(file_name):
        st.success(f"✅ FAISS index for `{file_name}` already exists.")
        return True

    # Create new FAISS index
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    vectorstore_faiss.save_local(index_name="index", folder_path=faiss_folder)

    # Upload FAISS Index & Metadata to S3
    s3_client.upload_file(faiss_index_path + ".faiss", BUCKET_NAME, f"faiss_files/{file_name}.faiss")
    s3_client.upload_file(pkl_path, BUCKET_NAME, f"faiss_files/{file_name}.pkl")

    st.success(f"✅ FAISS Index for `{file_name}` successfully created & uploaded!")

    return True

# List FAISS indexes from S3
def list_faiss_indexes():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    if "Contents" in response:
        return sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"] if obj["Key"].endswith(".faiss")))
    return []

# Load FAISS index from local storage
def load_faiss_index(index_name):
    """Loads FAISS index from local storage or downloads if missing."""
    if ensure_faiss_downloaded(index_name):
        return FAISS.load_local(
            index_name=index_name,
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    return None

# Initialize LLM
def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

# Extract document sources
def extract_source_documents(retrieved_docs):
    sources = set()
    for doc in retrieved_docs:
        if "source" in doc.metadata:
            sources.add(os.path.basename(doc.metadata["source"]))
    return sources

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

    # Upload PDFs
    st.subheader("Upload PDFs to Create Searchable Index")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_file_name = os.path.splitext(uploaded_file.name)[0]
            clean_name = clean_file_name(original_file_name)

            st.write(f"Processing PDF: {uploaded_file.name}")

            saved_file_name = os.path.join("/tmp", f"{clean_name}.pdf")
            with open(saved_file_name, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                loader = PyPDFLoader(saved_file_name)
                pages = loader.load_and_split()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue

            splitted_docs = split_text(pages)
            create_vector_store(clean_name, splitted_docs)

    # Question Answering
    st.subheader("Ask Questions from the Knowledge Base")

    faiss_indexes = list_faiss_indexes()
    if not faiss_indexes:
        st.error("No FAISS indexes found. Please upload PDFs first.")
        return

    selected_index = st.selectbox("Select a FAISS index", faiss_indexes)
    question = st.text_input(f"Ask a question about {selected_index}")

    if st.button("Ask Question"):
        selected_vectorstore = load_faiss_index(selected_index)
        response, retrieved_docs = get_response(get_llm(), selected_vectorstore, question)
        st.success("Here's the answer:")
        st.write(response["result"])

if __name__ == "__main__":
    main()
