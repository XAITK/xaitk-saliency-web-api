from fastapi import FastAPI, Header
from typing import Optional, Union, Sequence
from fastapi import FastAPI, UploadFile, File, Header
import uvicorn
import xaitk_saliency
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
import numpy as np
import PIL.Image
from fastapi.responses import FileResponse, Response, JSONResponse
from xaitk_saliency.utils.masking import occlude_image_batch
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import SimilarityScoring
from xaitk_saliency import PerturbImage, GenerateDescriptorSimilaritySaliency
from xaitk_saliency.utils.masking import occlude_image_batch
import os




app = FastAPI()


@app.post('/similarityScoring')
async def similarityScoring(loc: str, query_feat: UploadFile = File(...), ref_feat: UploadFile = File(...), pert_feat_ref: UploadFile = File(...), pert_masks: UploadFile = File(...)):
    query_feat = np.load(query_feat.file)
    ref_feat = np.load(ref_feat.file)
    pert_feat_ref = np.load(pert_feat_ref.file)
    pert_masks = np.load(pert_masks.file)

    similarity_alg = SimilarityScoring()

    salMaps = similarity_alg(query_feat,
                               ref_feat,
                               pert_feat_ref,
                               pert_masks)
    
    filepath = loc + '/' + 'sal_maps.npy'
    np.save(filepath, salMaps)

    return JSONResponse({'sal_maps_file': filepath})


@app.post("/occlusionMapFromFiles")
async def occlusionMapFromFiles(loc: str, npyfile: UploadFile = File(...), file: UploadFile = File(...), fill: UploadFile = File(...)):
    # masks = npyfile.file
    mask_array = np.load(npyfile.file, allow_pickle=True)
    fill = np.load(fill.file, allow_pickle=True)

    image = PIL.Image.open(file.file)
    ref_image = np.array(image)
    print(fill)
    occlusion_image_ndarray = occlude_image_batch(ref_image, mask_array, fill)

    filepath = loc + '/' + 'occlusionmap.npy'

    np.save(filepath, occlusion_image_ndarray)

    return JSONResponse({"occlusion_map_file": filepath})


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post('/perturb/')
async def create_upload_file(loc: str, file: UploadFile = File(...)):
    # open the image byte by byte

    img = PIL.Image.open(file.file)

    # throw image data into an array
    img_arr = np.array(img)

    # create the sliding window algorithm
    slid_algo = SlidingWindow(window_size=(40, 40), stride=(15, 15))

    # create multiple masks of the original image
    pert_masks = slid_algo.perturb(img_arr)

    filepath = loc + '/' + file.filename.split('.')[0] + '_pert_masks.npy'

    np.save(filepath, pert_masks)

    return JSONResponse(content = {'pert_masks_file': filepath})
    # return FileResponse(filename_to_be_created, filename=filename_to_be_created)
    # return Response(content = filename_to_be_created, media_type = '.npy')
    


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
async def displayImage(file: bytes = File(...)):
    # although media type is defined as image/jpeg this will work for other image formats as well
    # because the browser interprets the image regardless of the MIME type we define it as
    return Response(content=file, media_type='image/jpeg')


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)