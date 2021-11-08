from fastapi import FastAPI
from .main import app, create_upload_file, create_upload_file_with_parameters, displayImage, read_image_data, perturb_image, create_upload_file
from PIL import Image
from fastapi.testclient import TestClient
import requests

client = TestClient(app)

testimage = Image.open('1616031519513.jpg')
testimageurl = 'https://farm1.staticflickr.com/74/202734059_fcce636dcd_z.jpg'


def test_read_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_read_image_data():

    response = read_image_data(
        'https://farm1.staticflickr.com/74/202734059_fcce636dcd_z.jpg')
    assert response == {"Width": 640,
                        "Height": 480,
                        "Format": 'JPEG'}


def test_perturb_url():
    response = perturb_image(
        'https://farm1.staticflickr.com/74/202734059_fcce636dcd_z.jpg')
    assert response == {"Successfully perturbed": 200}


def test_create_upload_file():

    file = open('1616031519513.jpg', 'rb')
    response = create_upload_file(file)

    print(response)
    response == {"filename": "1616031519513.jpg",
                 "Pert Masks": [
                     1320,
                     574,
                     462
                 ]
                 }


def test_create_upload_with_parameters():

    file = open('1616031519513.jpg', 'rb')
    response = create_upload_file_with_parameters(40, 20, file)
    response == {"filename": "1616031519513.jpg",
                 "Pert Masks": [
                     920,
                     514,
                     362
                 ]
                 }


def test_display_image():
    file = open('1616031519513.jpg', 'rb')
    response = displayImage(file)
    response == {file}

