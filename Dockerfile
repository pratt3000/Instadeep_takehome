FROM pytorch/pytorch:latest

WORKDIR /code

ADD . /code

ENV PYTHONPATH "${PYTHONPATH}:/code"

RUN pip install -r requirements.txt