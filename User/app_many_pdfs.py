import boto3
import streamlit as st
import os
import uuid

# AWS S3 Configuration
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Explicitly Set AWS Region
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Initialize Bedrock Client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Use Correct Import for Bedrock Embeddings
from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Initialize Bedrock Embeddings (✅ Fixed)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

folder_path = "/tmp/"

# Function to list FAISS indexes in S3
def list_faiss_indexes():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="faiss_files/")
    if "Contents" in response:
        return sorted(set(obj["Key"].split("/")[-1].split(".")[0] for obj in response["Contents"]))
    return []

# Function to download a selected FAISS index from S3
def load_selected_index(selected_index):
    st.write(f"Loading FAISS Index: {selected_index}")
    s3_client.download_file(Bucket=BUCKET_NAME, Key=f"faiss_files/{selected_index}.faiss", Filename=f"{folder_path}{selected_index}.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key=f"faiss_files/{selected_index}.pkl", Filename=f"{folder_path}{selected_index}.pkl")

# Initialize the LLM
def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

# Retrieve answers using FAISS
def get_response(llm, vectorstore, question):
    st.write(f"Analyzing your question: {question}")

    # Define prompt
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

    # Retrieval-based QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": question})
    return answer['result']

# Main App
def main():
    st.header("Chat with Your PDF (RAG using Bedrock & FAISS)")

    # List available FAISS indexes
    faiss_indexes = list_faiss_indexes()

    if not faiss_indexes:
        st.error("No FAISS indexes found. Please upload PDFs in the Admin panel first.")
        return

    # Allow user to select an index
    selected_index = st.selectbox("Select a FAISS index", faiss_indexes)

    if st.button("Load Index"):
        load_selected_index(selected_index)
        st.success(f"Loaded index: {selected_index}")

    # Check if the selected index exists in local storage
    if os.path.exists(f"{folder_path}{selected_index}.faiss") and os.path.exists(f"{folder_path}{selected_index}.pkl"):
        st.write("Loading FAISS index into memory...")
        faiss_index = FAISS.load_local(
            index_name=selected_index,
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("FAISS Index is ready!")

        # Ask a Question
        question = st.text_input("Ask a question about your document")
        if st.button("Ask Question"):
            with st.spinner("Querying..."):
                llm = get_llm()
                answer = get_response(llm, faiss_index, question)
                st.write(answer)
                st.success("Done")
    else:
        st.warning("Please load an index first.")

if __name__ == "__main__":
    main()
