FROM python:3.12-slim-bullseye
WORKDIR /app
COPY . /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

EXPOSE 8502

CMD ["sh", "-c", "streamlit run streamlit_spam.py --server.enableCORS=false --server.address=0.0.0.0"]