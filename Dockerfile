FROM python:3.8-slim
RUN pip install --upgrade pip
WORKDIR /work
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn --host 0.0.0.0 --port 8000 docgen.server:app