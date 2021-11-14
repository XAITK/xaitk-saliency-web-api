from fastapi import FastAPI, Header
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header
import uvicorn
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
import numpy as np
import PIL.Image
from fastapi.responses import FileResponse, Response
from xaitk_saliency.utils.masking import occlude_image_batch
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import torch

from torchvision import transforms
from torch.autograd import Variable
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import SimilarityScoring
from torch import nn


from typing import Callable, Optional, Sequence, Union
from xaitk_saliency import PerturbImage, GenerateDescriptorSimilaritySaliency
from xaitk_saliency.utils.masking import occlude_image_batch

from xaitk_saliency.impls.perturb_image.sliding_radial import SlidingRadial
from scipy.ndimage import gaussian_filter


app = FastAPI()


@app.post("/occlusionMapFromFiles")
async def occlusionMapFromFiles(npyfile: UploadFile = File(...), file: UploadFile = File(...)):
    mask_array = np.load(npyfile.file)

    image = PIL.Image.open(file.file)
    ref_image = np.array(image)

    occlusion_image_ndarray = occlude_image_batch(ref_image, mask_array, None)
    np.save('occlusionmap.npy', occlusion_image_ndarray)
    print(len(occlusion_image_ndarray))
    return {"Made an occlusionmap": "found at [cwd/]occlusionmap.npy"}


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
async def displayImage(file: bytes = File(...)):
    # although media type is defined as image/jpeg this will work for other image formats as well
    # because the browser interprets the image regardless of the MIME type we define it as
    return Response(content=file, media_type='image/jpeg')


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)

#######Code for similarity scoring########


def similarityscoringapp(
    image_filepath_1: str,
    image_filepath_2: str,
    BlackboxModel: Callable[[np.ndarray], np.ndarray],
    perturber: PerturbImage,
    similarity_alg: GenerateDescriptorSimilaritySaliency,
    fill: Optional[Union[int, Sequence[int]]] = None,
    vis_mask_examples: bool = False,
):
    # Load the image
    query_image = np.array(PIL.Image.open(image_filepath_1))
    ref_image = np.array(PIL.Image.open(image_filepath_2))

    # Compute original feature vector on test images
    query_feat = BlackboxModel(query_image)
    ref_feat = BlackboxModel(ref_image)
    # Use the perturbation API implementation input to generate a bunch of images.
    # We will use the outputs here multiple times later so we will just aggregate
    # the output here.
    pert_masks = perturber(ref_image)
    print(f"perturbation masks: {pert_masks.shape}")

    # For the saliency heatmap generation API we need reference image feature vector as well as
    # the feature vectors for each of the perturbed reference images.
    pertbd_ref_imgs = occlude_image_batch(ref_image, pert_masks, fill)
    print(f"perturbed Reference images: {len(pertbd_ref_imgs)}")
    pert_feat_ref = np.asarray([
        BlackboxModel(pi)
        for pi in pertbd_ref_imgs
    ])
    # Visualize some example perturbed images before heading into similarity based saliency algorithm
    if vis_mask_examples:
        n = 4
        print(f"Visualizing {n} random perturbed reference images...")
        rng = np.random.default_rng(seed=0)
        rng_idx_lst = sorted(rng.integers(0, len(pert_masks)-1, n))
        plt.figure(figsize=(n*4, 3))
        for i, rnd_i in enumerate(rng_idx_lst):
            plt.subplot(1, n, i+1)
            plt.title(f"pert_imgs[{rnd_i}]")
            plt.axis('off')
            plt.imshow(pertbd_ref_imgs[rnd_i])

    print(f"Pert features: {pert_feat_ref.shape}")

    # Generating final similarity based saliency map
    sal_maps = similarity_alg(query_feat,
                              ref_feat,
                              pert_feat_ref,
                              pert_masks)

    sub_plot_ind = len(sal_maps) + 2
    plt.figure(figsize=(12, 6))
    plt.subplot(2, sub_plot_ind, 1)
    plt.imshow(query_image)
    plt.axis('off')
    plt.title('Query Image')

    plt.subplot(2, sub_plot_ind, 2)
    plt.imshow(ref_image)
    plt.axis('off')
    plt.title('Reference Image')

    # Some magic numbers here to get colorbar to be roughly the same height
    # as the plotted image.
    colorbar_kwargs = {
        "fraction": 0.046*(ref_image.shape[1]/ref_image.shape[0]),
        "pad": 0.04,
    }

    for i, class_sal_map in enumerate(sal_maps):
        print(
            f"Reference saliency map range: [{class_sal_map.min()}, {class_sal_map.max()}]")

        # Positive half saliency
        plt.subplot(2, sub_plot_ind, 3+i)
        plt.imshow(ref_image, alpha=0.7)
        plt.imshow(
            np.clip(class_sal_map, 0, 1),
            cmap='jet', alpha=0.3
        )
        plt.clim(0, 1)
        plt.colorbar(**colorbar_kwargs)
        plt.title(f"Reference Image #{i+1} Pos Saliency")
        plt.axis('off')

        # Negative half saliency
        plt.subplot(2, sub_plot_ind, sub_plot_ind+3+i)
        plt.imshow(ref_image, alpha=0.7)
        plt.imshow(
            np.clip(class_sal_map, -1, 0),
            cmap='jet_r', alpha=0.3
        )
        plt.clim(-1, 0)
        plt.colorbar(**colorbar_kwargs)
        plt.title(f"Reference Image #{i+1} Neg Saliency")
        plt.axis('off')
    plt.show()
    plt.close()


CUDA_AVAILABLE = torch.cuda.is_available()

# Creating an example deep feature extractor to describe images
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.eval()

if CUDA_AVAILABLE:
    model = model.cuda()

################4####################


class SimilarityBlackbox():

    def __init__(self,
                 model):
        self.model = model
        self.model_input_size = (224, 224)
        self.model_mean = [0.485, 0.456, 0.406]
        self.blackbox_fill = np.uint8(np.asarray(self.model_mean) * 255)
        self.model_loader = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model_mean,
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def image_loader(self, image):
        image = self.model_loader(image).float()
        image = Variable(image, requires_grad=False)
        if CUDA_AVAILABLE:
            image = image.cuda()
        return image.unsqueeze(0)

    @torch.no_grad()
    def __call__(self, image):
        featureVec = self.model(self.image_loader(image))
        return featureVec.cpu().detach().numpy().squeeze()


simbbox = SimilarityBlackbox(model)


@app.post("/SimilarityScoringDefault")
async def similarity_scoring_default(query: UploadFile = File(...), reference: UploadFile = File(...)):
    slid_algo = SlidingWindow(window_size=(40, 40), stride=(15, 15))

    similarity_alg = SimilarityScoring()

    sal_maps = similarityscoringapp(
        query.file,
        reference.file,
        simbbox,
        slid_algo,
        similarity_alg,
        fill=simbbox.blackbox_fill,
        vis_mask_examples=True,
    )
    return {"Successfully Perturbed": 200}


@app.post('/RadialImagePerturbation')
async def radial_image_perturbation(image: UploadFile = File(...)):
    gh_mat = np.asarray(PIL.Image.open(image.file))
    gh_mat_blur = gaussian_filter(gh_mat, sigma=(20, 20, 0))

    # Let's take a look!
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.axis(False)
    plt.imshow(gh_mat)

    plt.subplot(1, 2, 2)
    plt.axis(False)
    plt.imshow(gh_mat_blur)
    plt.show()
    plt.close()
    masks = SlidingRadial(radius=(125, 125), stride=[
                          200, 200], sigma=(20, 20)).perturb(gh_mat)
    # print(masks.shape)

    idx = 4

    # apply one mask to the image, with blurred-image alpha blending.
    gh_rad_occ = occlude_image_batch(gh_mat, masks[None, idx], gh_mat_blur)[0]

    # Display the mask and the result blended image.
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title("Mask")
    plt.axis(False)
    plt.imshow(masks[idx], vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Occlusion Result")
    plt.axis(False)
    plt.imshow(gh_rad_occ)
    plt.show()
    plt.close()
    return {"Successfully Perturbed": 200}
