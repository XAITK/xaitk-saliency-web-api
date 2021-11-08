from typing import Optional
from fastapi import FastAPI, UploadFile, File
import uvicorn
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
import numpy as np
import PIL.Image
import urllib.request
from fastapi.responses import FileResponse, Response

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}

#gets an image by url and returns basic data (dimensions and format)
@app.get("/imageurl")
def read_image_data(image_url: str):
    try:
       image = PIL.Image.open(urllib.request.urlopen(image_url))

       return {"Width": image.size[0], 
            "Height": image.size[1], 
            "Format": image.format}
    except:
        #there was an error getting the image
        return {"Error": "incorrecturl"}

#perturbs the image using the xaitk sliding window impl
@app.get("/perturb/sliding_window")
def perturb_image(image_url: str):
    #takes in an image from url and perturbs it, storing resulting masks 
    #in a ndarray of booleans (this will in turn be used in another call)
    try:
        #Getting the image
        image = PIL.Image.open(urllib.request.urlopen(image_url))
        #turning into numpy array, 3D array of ints
        ref_image = np.array(image)
        #creating a sliding window perturbation class
        slidingwindow = SlidingWindow()
        #perturbing the image, saving as 'perturbeddataarray'
        perturbeddataarray = slidingwindow.perturb(ref_image)
       
        return {"Successfully perturbed": 200}
        
    except:
        #error getting the image
        return {"ERROR": "url"}

@app.post('/perturb')
async def create_upload_file(file: UploadFile = File(...)):
    # open the image byte by byte
    img = PIL.Image.open(file.file)

    # throw image data into an array
    img_arr = np.array(img)

    # create the sliding window algorithm 
    slid_algo = SlidingWindow(window_size=(40, 40), stride=(15, 15))

    # create multiple masks of the original image 
    pert_masks = slid_algo.perturb(img_arr)

    return { 
            
            "filename": file.filename,
            "Pert Masks": pert_masks.shape

    }

# takes in an image as input and returns the image as output
@app.post('/displayImage')
# image file needs to be uploaded as bytes for it to be returned properly
async def goPup(file: bytes = File(...)):
    # although media type is defined as image/jpeg this will work for other image formats as well
    # because the browser interprets the image regardless of the MIME type we define it as
    return Response(content = file, media_type= 'image/jpeg')

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)