from fastapi import FastAPI, Depends
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header
from smqtk_classifier.interfaces.classify_image import ClassifyImage
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
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import SlidingWindowStack
from xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring import OcclusionScoring
import pickle



app = FastAPI()

@app.post('/OcclusionScoring')
async def occlusionScoring(ref_preds: bytes = File(...), pert_preds: bytes = File(...), pert_masks: bytes = File(...)):
    # convert ref_preds, pert_preds, and pert_masks from bytes to ndarrays
    ref_preds = pickle.loads(ref_preds)
    pert_preds = pickle.loads(pert_preds)
    pert_masks = pickle.loads(pert_masks)

    # initialize OcclusionScoring class
    sal_maps_generator = OcclusionScoring()

    # generate saliency maps using OcclusionScoring
    sal_maps = sal_maps_generator(ref_preds, pert_preds, pert_masks)

    # convert saliency maps from ndarray to bytes
    sal_maps = pickle.dumps(sal_maps)
    
    # return saliency maps as bytes
    return Response(content = sal_maps)

# not functional yet
# need to figure out how to pass a blackbox model through the API
@app.post('/SlidingWindowStack')
async def slidingWindowStack(window_height: int, window_width: int, stride_height_step: int, stride_width_step: int, num_threads: Optional[int],  
                             fill: Optional[bytes] = File(...), ref_image: UploadFile = File(...), blackbox: bytes = File(...)):

    # load image from SpooledTemoraryFile to ndarray
    image = PIL.Image.open(ref_image.file)
    ref_image = np.array(image)

    # load fill from bytes to ndarray
    fill = pickle.loads(fill)

    # load blackbox classifier from bytes 
    blackbox = pickle.loads(blackbox)

    # initialize SlidingWindowStack with params and apply fill
    gen_sliding_window = SlidingWindowStack((window_height, window_width), (stride_height_step, stride_width_step), threads=num_threads)
    gen_sliding_window.fill = fill

    # generate saliency maps from ref image and blackbox algorithm
    sal_maps = gen_sliding_window(ref_image, blackbox)

    # convert sal_maps from ndarray to bytes
    sal_maps_as_bytes = pickle.dumps(sal_maps)

    return Response(content = sal_maps_as_bytes)


@app.post('/similarityScoring')
async def similarityScoring(query_feat: bytes = File(...), ref_feat: bytes = File(...), pert_feat_ref: bytes = File(...), pert_masks: bytes = File(...)):
    # convert all byte arrays to ndarrays
    query_feat = pickle.loads(query_feat)
    ref_feat = pickle.loads(ref_feat)
    pert_feat_ref = pickle.loads(pert_feat_ref)
    pert_masks = pickle.loads(pert_masks)

    # initialize Similarity Scoring Algorithm
    similarity_alg = SimilarityScoring()

    # save saliency ndarray to sal_maps
    sal_maps = similarity_alg(query_feat,
                               ref_feat,
                               pert_feat_ref,
                               pert_masks)
    
    # convert sal_maps from ndarray to bytes 
    sal_maps_as_bytes = pickle.dumps(sal_maps)

    return Response(content = sal_maps_as_bytes)


@app.post("/occlude_image_batch")
async def occlusionMapFromFiles(pert_masks: bytes = File(...), fill: Optional[bytes] = File(...), ref_image: UploadFile = File(...)):
    # load pert_masks from bytes to ndarray
    pert_masks = pickle.loads(pert_masks)

    # load image from SpooledTemoraryFile to ndarray
    image = PIL.Image.open(ref_image.file)
    ref_image = np.array(image)

    # if there is a fill array load it from bytes to ndarray
    if fill:
        fill = pickle.loads(fill)

        # run xaitk occlude_image_batch with fill
        occlusion_image_ndarray = occlude_image_batch(ref_image, pert_masks, fill)

        # convert ndarray to bytes
        occlusion_image_as_bytes = pickle.dumps(occlusion_image_ndarray)

        return Response(content = occlusion_image_as_bytes)
    
    # run xaitk occlude_image_batch without fill
    occlusion_image_ndarray = occlude_image_batch(ref_image, pert_masks)

    # convert ndarray to bytes
    occlusion_image_as_bytes = pickle.dumps(occlusion_image_ndarray)

    return Response(content = occlusion_image_as_bytes)

@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post('/sliding_window_perturb/')
async def create_upload_file(window_size_x: int, window_size_y: int, stride_x: int, stride_y: int, ref_image: UploadFile = File(...)):
    # open the image byte by byte
    img = PIL.Image.open(ref_image.file)

    # throw image data into an array
    img_arr = np.array(img)

    # create the sliding window algorithm
    slid_algo = SlidingWindow(window_size=(window_size_x, window_size_y), stride=(stride_x, stride_y))

    # create multiple masks of the original image
    pert_masks = slid_algo.perturb(img_arr)

    pert_masks_as_bytes = pickle.dumps(pert_masks)
    
    return Response(content = pert_masks_as_bytes)
    


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)