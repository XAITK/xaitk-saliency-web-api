from fastapi import FastAPI, Header
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import uvicorn
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
import numpy as np
import PIL.Image
from fastapi.responses import FileResponse, Response


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


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

    filename_to_be_created = '' + file.filename + '_pert_masks.npy'
    np.save(filename_to_be_created, pert_masks)

    return FileResponse(filename_to_be_created, filename=filename_to_be_created)


@app.post('/perturbWithParameters')
async def create_upload_file_with_parameters(windowSizeDefault40, windowStrideDefault15, file: UploadFile = File(...)):
    # open the image byte by byte
    img = PIL.Image.open(file.file)

    # throw image data into an array
    img_arr = np.array(img)

    # create the sliding window algorithm
    slid_algo = SlidingWindow(window_size=(windowSizeDefault40, windowSizeDefault40), stride=(
        windowStrideDefault15, windowStrideDefault15))

    # create multiple masks of the original image
    pert_masks = slid_algo.perturb(img_arr)
    filename_to_be_created = '' + file.filename + '_pert_masks.npy'
    np.save(filename_to_be_created, pert_masks)

    return FileResponse(filename_to_be_created, filename=filename_to_be_created)


@app.post('/displayImage')
# image file needs to be uploaded as bytes for it to be returned properly
async def displayImage(file: bytes = File(...)):
    # although media type is defined as image/jpeg this will work for other image formats as well
    # because the browser interprets the image regardless of the MIME type we define it as
    return Response(content=file, media_type='image/jpeg')


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)

