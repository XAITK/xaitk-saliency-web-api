# XAITK - Saliency

## Intent
Provide dockerized api module for interaction with https://github.com/XAITK/xaitk-saliency package.

## Getting started

### Prerequisites:
1. Install the latest version of Python.
2. Run `pip install packagename` for the following packages:
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; fastapi <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  pydantic <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; uvicorn <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; click <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; h11 <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; starlette <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; xaitk-saliency <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; python-multipart <br>

3. Install Docker Desktop.

```
Note: If any new packages are installed then the requirements.txt file needs to be updated with the name and version # of the installed package.
```

### Installing
4. Open an IDE.
5. Connect to the Github repository for the project: https://github.com/XAITK/xaitk-saliency-web-api and pull the latest code.

### Running locally
6. To run the code on a local server navigate to the app directory and execute the following command from the terminal: 
```
python -m uvicorn main:app --reload
``` 
7. Upon execution Uvicorn will run the code on a local server and return the link to it in the terminal. Following the link will open the / endpoint by default. To view the SwaggerUI documentation page append /docs to the URL in your browser.


### Warranty
Code was last updated on 11/14/21

## Testing

To run the test suite run the following command from the terminal: 

``` 
python -m pytest
```
The results of the tests will return in the terminal.

## Deployment
This app is deployed through Docker and is distributed through DockerHub via this command:
```
docker pull robertjb/xaitkimage
```
### End users can use the app by following these steps:
Open command line and execute the following commands:
```
- docker pull robertjb/xaitkimage
- docker run -p 8000:8000 --name xaitkcontainer robertjb/xaitkimage
```
- The container name and port number are arbitrary.

Once these commands are run the user can access the API with the this link: `localhost:<port#>/docs` or by via a HTTP request to: `localhost:<port#>/<endpoint>`.


### Developers can use the app by following these steps:
Docker container build:
```
 $ docker build -t xaitkimage .
```

Docker container run:
```
 $ docker run -p 8000:8000 --name xaitkcontainer xaitkimage
```

Access container:
```
 localhost:8000/docs
```

Kill container
```
 $ docker kill xaitkcontainer
```

Launch Swagger UI
- http://localhost:8000/docs

```
Note: You will need to rebuild the Docker container after making changes to the API.
```

## Technologies Used

### Docker
For this project we used Docker to "host" our application, essentially the Docker image's purpose is to build the local API environment for the user without them having to install any dependancies.

### FastAPI & Xaitk-Saliency
This app is built entirely within the FastAPI framework and most of the code within each endpoint is a part of the xaitk-saliency package.

### Architecture
Our architecture diagram can be found here: https://github.com/XAITK/xaitk-saliency-web-api/blob/master/architecture_diagram.pdf


## Contributing
This is an open-source project. Contributors have access to everything they need to work on this project via the Github link. For notes on style, testing, and process conventions please reference the project website here: https://tarheels.live/xaitkprojectportfolio/team/.

## Authors
### Xaitk-Saliency Web API
Zack Zeplin

Griffin Groh

Robert Bennett

### Xaitk-Saliency
Kitware

## License
?

## API Documentation

>### /imageurl
- input: image URL
- output: {image height, 
           image width, 
           image format
        }

>### /perturb/sliding_window
- input: image URL
- output: 200 Success or Error

>### /perturb
- input: image file
- output: .npy file containing the matrix of perturbation masks for the image

>### /perturbWithParameters
- input: windowSize, windowStride, image file
- output: .npy file containing the matrix of perturbation masks for the image

>### /displayImage
- input: image file
- output: image file
