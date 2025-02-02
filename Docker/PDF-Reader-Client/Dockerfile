FROM python:3.11
EXPOSE 8084
WORKDIR /app
COPY ./User/requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8084" ]
