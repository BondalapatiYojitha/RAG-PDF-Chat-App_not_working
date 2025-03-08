import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
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

# Fetch FAISS indexes from S3 and download them locally
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

# Load FAISS index from local storage
def load_faiss_index(index_name):
    """Load a FAISS index from the local folder."""
    try:
        return FAISS.load_local(
            index_name=index_name,
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading FAISS index {index_name}: {e}")
        return None

# Merge all FAISS indexes into a single vectorstore
def merge_all_indexes(excluded_index=None):
    """Merge all FAISS indexes into a single searchable vectorstore."""
    all_faiss_indexes = list_faiss_indexes()
    combined_vectorstore = None

    for index in all_faiss_indexes:
        if index == excluded_index:
            continue  # Skip the already searched document

        faiss_index = load_faiss_index(index)

        if faiss_index:
            if combined_vectorstore is None:
                combined_vectorstore = faiss_index
            else:
                combined_vectorstore.merge_from(faiss_index)

    return combined_vectorstore

# Initialize LLM
def get_llm():
    """Get a Bedrock LLM instance."""
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

# Perform retrieval and generate response
def get_response(llm, vectorstore, question):
    """Retrieve answers using FAISS and display source documents."""
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

# Extract source document names
def extract_source_documents(retrieved_docs):
    """Extracts and formats source document names from retrieved docs."""
    sources = set()
    for doc in retrieved_docs:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])

    return sources

# Main Streamlit App
def main():
    st.title("Chat with Your PDF")

    # Question Answering Section
    st.subheader("Ask Questions from the Knowledge Base")

    faiss_indexes = list_faiss_indexes()
    if not faiss_indexes:
        st.error("No FAISS indexes found in S3. Please upload PDFs first.")
        return

    selected_index = st.selectbox("Select a FAISS index", faiss_indexes)
    question = st.text_input(f"Ask a question about {selected_index}")

    if st.button("Ask Question"):
        with st.spinner("Finding the best answer..."):
            # Step 1: Search in Selected Document
            selected_vectorstore = load_faiss_index(selected_index)

            if selected_vectorstore:
                response, retrieved_docs = get_response(get_llm(), selected_vectorstore, question)

                if retrieved_docs:
                    sources = extract_source_documents(retrieved_docs)
                    source_text = f"\n\n📄 Answer found in: {', '.join(sources) if sources else selected_index}"
                    st.success("Here's the answer:")
                    st.write(response["result"] + source_text)
                    return  # ✅ Exit since we found the answer in the selected document

            # Step 2: Search in All Documents
            st.warning("Answer not found in selected document. Searching all indexes...")
            all_vectorstore = merge_all_indexes(excluded_index=selected_index)

            if all_vectorstore:
                response, retrieved_docs = get_response(get_llm(), all_vectorstore, question)

                if retrieved_docs:
                    sources = extract_source_documents(retrieved_docs)
                    source_text = f"\n\n📄 Answer found in: {', '.join(sources) if sources else 'Unknown Source'}"
                    st.success("Here's the answer:")
                    st.write(response["result"] + source_text)
                    return

            # Step 3: No Answer Found
            st.error(f"❌ Couldn't find relevant information in {selected_index} or any other document.")

if __name__ == "__main__":
    main()
