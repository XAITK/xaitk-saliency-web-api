# XAITK - Saliency

## Intent
Provide dockerized api module for interaction with https://github.com/XAITK/xaitk-saliency package

## Documentation

## Setup
Docker container build:
- $ docker build -t xaitkimage .

Docker container run:
- $ docker run -p 8000:8000 --name xaitkcontainer xaitkimage

Access container:
- localhost:8000/docs

Kill container
- $ docker kill xaitkcontainer