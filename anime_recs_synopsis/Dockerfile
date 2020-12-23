FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . /app