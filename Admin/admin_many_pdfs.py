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

# Create FAISS Index
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
    return FAISS.load_local(
        index_name=index_name,
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

# Merge all FAISS indexes into a single vectorstore
def merge_all_indexes(excluded_index=None):
    all_faiss_indexes = list_faiss_indexes()
    combined_vectorstore = None

    for index in all_faiss_indexes:
        if index == excluded_index:
            continue  

        faiss_index = load_faiss_index(index)
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

# Extract document sources
def extract_source_documents(retrieved_docs):
    sources = set()
    for doc in retrieved_docs:
        if "source" in doc.metadata:
            sources.add(os.path.basename(doc.metadata["source"]))
    return sources

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
            st.write(f"Creating the Vector Store for {uploaded_file.name}...")
            create_vector_store(clean_name, splitted_docs)
            st.success(f"Successfully processed {uploaded_file.name}!")

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
            response, retrieved_docs = get_response(get_llm(), selected_vectorstore, question)

            if retrieved_docs:
                sources = extract_source_documents(retrieved_docs)
                source_text = f"\n\n📄 Answer found in: {', '.join(sources) if sources else selected_index}"
                st.success("Here's the answer:")
                st.write(response["result"] + source_text)
                return

            st.error(f"❌ Couldn't find relevant information in {selected_index} or any other document.")

if __name__ == "__main__":
    main()
