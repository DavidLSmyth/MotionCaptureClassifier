FROM python:3.8.0-slim as builder
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

COPY docker_requirements.txt docker_requirements.txt
COPY . /


WORKDIR /
RUN pip install --upgrade pip
RUN pip install -r docker_requirements.txt
CMD python API/server/LSTMClassificationAPIServer.py