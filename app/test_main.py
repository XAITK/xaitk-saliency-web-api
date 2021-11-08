from fastapi import FastAPI
from .main import app
from PIL import Image
from fastapi.testclient import TestClient

client = TestClient(app)

testimage = Image.open('test.jpeg')
testimageurl = 'https://farm1.staticflickr.com/74/202734059_fcce636dcd_z.jpg'

# works


def test_read_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_read_image_data():
    data = {"image_url": 'https://farm1.staticflickr.com/74/202734059_fcce636dcd_z.jpg'}
    response = client.get(
        "/imageurl", "https://farm1.staticflickr.com/74/202734059_fcce636dcd_z.jpg")
    assert response.status_code == 200

    # 422 error?


def test_create_upload_file():
    with open("borat.jpeg", "rb") as image:
        f = image.read()
        b = bytearray(f)

    response = client.post('/perturb', b)
    assert response.status_code == 200

    # def test_read_image():
    #    response = client.get("/imageurl", headers={""})
    #    assert response.status_code == 200
