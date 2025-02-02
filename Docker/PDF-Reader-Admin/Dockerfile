FROM python:3.11
EXPOSE 8083
WORKDIR /app
COPY ./Admin/requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "admin.py", "--server.port=8083" ]
