# XAITK - Saliency

## Intent
Provide dockerized api module for interaction with https://github.com/XAITK/xaitk-saliency package

## Setup
Docker container build:
- $ docker build -t xaitkimage .

Docker container run:
- $ docker run -p 8000:8000 --name xaitkcontainer xaitkimage

Access container:
- localhost:8000/docs

Kill container
- $ docker kill xaitkcontainer

Launch Swagger UI
- http://localhost:8000/docs

## Documentation

/imageurl
- input: image URL
- output: {image height, 
           image width, 
           image format
        }

/perturb/sliding_window
- input: image URL
- output: 200 Success or Error

/perturb
- input: image file
- output: {filename, shape of the perturbation masks array after image perturbation}