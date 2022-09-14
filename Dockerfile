FROM python:3.9.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD ["sh", "-c", "streamlit run --server.port $PORT app.py"]