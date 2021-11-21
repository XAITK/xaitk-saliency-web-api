import requests
import matplotlib
import os
import torchvision.models as models
import torch
from torch import nn
import numpy as np
import PIL.Image
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import SimilarityScoring
import matplotlib.pyplot as plt


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

###################################################

query_image = np.array(PIL.Image.open('catDog.jpg'))
ref_image = np.array(PIL.Image.open('surfDog.jpg'))

# Compute original feature vector on test images
query_feat = simbbox(query_image)
ref_feat = simbbox(ref_image)

directory = {'loc': os.getcwd()}
ref_img_json = {'file': open('surfDog.jpg', 'rb')}

pert_masks = requests.post('http://127.0.0.1:8000/perturb/', files=ref_img_json, params = directory).json()
pert_masks = pert_masks['pert_masks_file']
print(pert_masks)

np.save('fill.npy', simbbox.blackbox_fill)
masksRefImg = {'npyfile': open(pert_masks, 'rb'), 'file': open('surfDog.jpg', 'rb'), 'fill': open('fill.npy', 'rb')}

occMap = requests.post('http://127.0.0.1:8000/occlusionMapFromFiles', files = masksRefImg, params = directory).json()
occMap = occMap['occlusion_map_file']
print(occMap)

occMap = np.load(occMap)

pert_feat_ref = np.asarray([
        simbbox(pi)
        for pi in occMap
    ])

np.save('pert_feat_ref.npy', pert_feat_ref)
np.save('query_feat.npy', query_feat)
np.save('ref_feat.npy', ref_feat)

sim_scoring_files = {'query_feat': open('query_feat.npy', 'rb'), 'ref_feat': open('ref_feat.npy', 'rb'), 'pert_feat_ref': open('pert_feat_ref.npy', 'rb'), 'pert_masks': open(pert_masks, 'rb')}

sal_maps = requests.post('http://127.0.0.1:8000/similarityScoring', files=sim_scoring_files, params = directory).json()
sal_maps = sal_maps['sal_maps_file']
print(sal_maps)
sal_maps = np.load(sal_maps)

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
