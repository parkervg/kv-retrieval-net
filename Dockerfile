FROM python:3.8
EXPOSE $PORT
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

RUN mkdir -p src
RUN mkdir -p resources
ADD src src
ADD resources resources

WORKDIR src

# Use $PORT so Heroku can deploy it correctly
CMD uvicorn --host 0.0.0.0 --port $PORT main:app
