FROM python:3.11
EXPOSE 8083
WORKDIR /app
<<<<<<< HEAD
COPY ./Admin/requirements.txt ./
RUN pip install -r requirements.txt
=======
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
>>>>>>> 3ecd1539bc53e705832ea78d9139be509c5e26c3
COPY . ./
ENTRYPOINT [ "streamlit", "run", "admin.py", "--server.port=8083" ]
