FROM python:3.11

# Expose the port for Streamlit
EXPOSE 8084

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY ./User/requirements.txt ./
RUN pip install -r requirements.txt

# Copy the entire project into the container
COPY . ./

# Accept AWS credentials as build arguments
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION

# Set AWS credentials as environment variables inside the container
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_REGION=${AWS_REGION}

# Debugging: Print environment variables inside the container
RUN echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" && \
    echo "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" && \
    echo "AWS_REGION=$AWS_REGION"

# Start Streamlit app
ENTRYPOINT [ "streamlit", "run", "./User/app_many_pdfs.py", "--server.port=8084", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false" ]
