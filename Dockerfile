FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

RUN mkdir -p src
RUN mkdir -p resources
RUN mkdir -p data
ADD src src
ADD resources resources
ADD data data

WORKDIR src
# Use $PORT so Heroku can deploy it correctly
CMD uvicorn --host 0.0.0.0 --port $PORT main:app
