FROM python:3.9.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
ENTRYPOINT ["streamlit", "run"]
CMD gunicorn --workers=4 --bind:0.0.0.0:$PORT app:app