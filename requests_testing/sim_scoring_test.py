import requests
import torchvision.models as models
import torch
from torch import nn
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import pickle


CUDA_AVAILABLE = torch.cuda.is_available()

# Creating an example deep feature extractor to describe images
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.eval()

if CUDA_AVAILABLE:
    model = model.cuda()

from torch.autograd import Variable
from torchvision import transforms


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

#######################################################################################

query_image = np.array(PIL.Image.open('./requests_testing/catDog.jpg'))
ref_image = np.array(PIL.Image.open('./requests_testing/surfDog.jpg'))

# Compute original feature vector on test images
query_feat = simbbox(query_image)
ref_feat = simbbox(ref_image)

# convert query and reference features from ndarray to bytes 
query_feat = pickle.dumps(query_feat)
ref_feat = pickle.dumps(ref_feat)

# create json query params for the sliding window perturb API call
sliding_window_params = {'window_size_x': 40, 'window_size_y': 40, 'stride_x': 15, 'stride_y': 15}
ref_img_file = {'ref_image': open('./requests_testing/surfDog.jpg', 'rb')}

# generate perturbation masks via xaitk_web_api
pert_masks = requests.post('http://127.0.0.1:8000/sliding_window_perturb/', files=ref_img_file, params = sliding_window_params).content

# convert fill ndarray to bytes obj
fill = pickle.dumps(simbbox.blackbox_fill)

# create json query params for the occlude_image_batch API call
occlude_image_params = {'pert_masks': pert_masks, 'fill': fill, 'ref_image': open('./requests_testing/surfDog.jpg', 'rb')}

# generate occluded image batch via xaitk_web_api
occMap = requests.post('http://127.0.0.1:8000/occlude_image_batch', files = occlude_image_params).content
occMap = pickle.loads(occMap)

# use occlusion map to generate feature scores for each perturbed image
pert_feat_ref = np.asarray([
        simbbox(pi)
        for pi in occMap
    ])

# convert feature scores from ndarray to bytes
pert_feat_ref = pickle.dumps(pert_feat_ref)

# create json query params for the Similarity Scoring API call
sim_scoring_params = {'query_feat': query_feat, 'ref_feat': ref_feat, 'pert_feat_ref': pert_feat_ref, 'pert_masks': pert_masks}

# use the web api to generate a saliency ndarray 
sal_maps = requests.post('http://127.0.0.1:8000/similarityScoring', files=sim_scoring_params).content

# convert the saliency maps from bytes to a ndarray
sal_maps = pickle.loads(sal_maps)

# plot the saliency maps with the query and ref image
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
    print(f"Reference saliency map range: [{class_sal_map.min()}, {class_sal_map.max()}]")
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









