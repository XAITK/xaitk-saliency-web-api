# Set up our "black box" classifier using PyTorch and it's ImageNet pretrained ResNet18.
# We will constrain the output of our classifier here to the two classes that are relevant
# to our test image for the purposes of this example.
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image
import matplotlib.pyplot as plt
import requests
import pickle

from xaitk_saliency.utils.masking import occlude_image_batch

CUDA_AVAILABLE = torch.cuda.is_available()


model = models.resnet18(pretrained=True)
model = model.eval()
if CUDA_AVAILABLE:
    model = model.cuda()

# These are some simple helper functions to perform prediction with this model
model_input_size = (224, 224)
model_mean = [0.485, 0.456, 0.406]
model_loader = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(model_input_size), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model_mean,
        std=[0.229, 0.224, 0.225]
    ),
])


# Grabbing the class labels associated with this model.
classes_file = "requests_testing/imagenet_classes.txt"
# if not os.path.isfile(classes_file):
#     !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt -O {classes_file}

f = open(classes_file, "r")
categories = [s.strip() for s in f.readlines()]

# For this test, we will use an image with both a cat and a dog in it.
# Let's only consider the saliency of two class predictions.
sal_class_labels = ['boxer', 'tiger cat']
sal_class_idxs = [categories.index(lbl) for lbl in sal_class_labels]


@torch.no_grad()
def blackbox_classifier(test_image: np.ndarray) -> np.ndarray:
    image_tensor = model_loader(test_image).unsqueeze(0)
    if CUDA_AVAILABLE:
        image_tensor = image_tensor.cuda()
    feature_vec = model(image_tensor)
    # Converting feature extractor output to probabilities.
    class_conf = torch.nn.functional.softmax(feature_vec, dim=1).cpu().detach().numpy().squeeze()
    # Only return the confidences for the focus classes
    return class_conf[sal_class_idxs]


blackbox_fill = np.uint8(np.asarray(model_mean) * 255)

# Make use of superpixel based mask generation
from skimage.segmentation import quickshift, mark_boundaries

# Load the reference image
ref_image = PIL.Image.open('requests_testing/catDog.jpg')
# Generate superpixel segments
segments = quickshift(ref_image, kernel_size=4, max_dist=200, ratio=0.2, random_seed=0)
# Print number of segments
num_segments = len(np.unique(segments))
print("Quickshift number of segments: {}".format(num_segments))


# Visualize the superpixels on the image
plt.figure(figsize=(12, 8))
plt.axis('off')
_ = plt.imshow(mark_boundaries(ref_image, segments))
# plt.show()

# Next, we'll convert these superpixel segments to binary perturbation masks in preparation for generating the corresponding perturbation images.
pert_masks = np.empty((num_segments, *ref_image.size[::-1]), dtype=bool)
for i in range(num_segments):
    pert_masks[i] = (segments != i)


# Load the image
ref_image = np.asarray(PIL.Image.open('requests_testing/catDog.jpg'))

print(pert_masks.shape)

# convert fill and pert_masks to bytes
blackbox_fill = pickle.dumps(blackbox_fill)
pert_masks = pickle.dumps(pert_masks)

# set api params
occlude_image_batch_files = {'pert_masks': pert_masks, 'fill': blackbox_fill, 'ref_image': open('./requests_testing/catDog.jpg', 'rb')}

# Remember that we defined our own perturbation masks, and will
# now use a helper function to generate perturbation images
pert_imgs = requests.post('http://127.0.0.1:8000/occlude_image_batch', files = occlude_image_batch_files).content

# convert pert_imgs to ndarray
pert_imgs = pickle.loads(pert_imgs)
print(f"Perterbed images: {pert_imgs.shape[0]}")


# n = 4
# print(f"Visualizing {n} random perturbed images...")

# rng = np.random.default_rng(seed=0)
# rng_idx_lst = sorted(rng.integers(0, len(pert_imgs)-1, n))
# plt.figure(figsize=(n*4, 4))

# for i, rnd_i in enumerate(rng_idx_lst):
#     plt.subplot(1, n, i+1)
#     plt.title(f"pert_imgs[{rnd_i}]")
#     plt.axis('off')
#     plt.imshow(pert_imgs[rnd_i])

# For the saliency heatmap generation API we need reference image predictions as well as
# the predictions for each of the perturbed images.
ref_preds = blackbox_classifier(ref_image)
print(f"Ref preds: {ref_preds.shape}")
pert_preds = np.asarray([
        blackbox_classifier(pi)
        for pi in pert_imgs
])

print(f"Pert preds: {pert_preds.shape}")
print(type(ref_preds))
print(type(pert_preds))
print(type(pert_masks))
ref_preds = pickle.dumps(ref_preds)
pert_preds = pickle.dumps(pert_preds)
# pert_masks = pickle.dumps(pert_masks)

occlusion_scoring_files = {'ref_preds': ref_preds, 'pert_preds': pert_preds, 'pert_masks': pert_masks}

sal_maps = requests.post('http://127.0.0.1:8000/OcclusionScoring/', files = occlusion_scoring_files).content

sal_maps = pickle.loads(sal_maps)

print(f"Saliency maps: {sal_maps.shape}")

# Visualize the saliency heat-maps
sub_plot_ind = len(sal_maps) + 1
plt.figure(figsize=(12, 6))
plt.subplot(2, sub_plot_ind, 1)
plt.imshow(ref_image)
plt.axis('off')
plt.title('Test Image')
# Some magic numbers here to get colorbar to be roughly the same height
# as the plotted image.
colorbar_kwargs = {
    "fraction": 0.046*(ref_image.shape[0]/ref_image.shape[1]),
    "pad": 0.04,
}
for i, class_sal_map in enumerate(sal_maps):
    print(f"Class {i} saliency map range: [{class_sal_map.min()}, {class_sal_map.max()}]")
    # Positive half saliency
    plt.subplot(2, sub_plot_ind, 2+i)
    plt.imshow(ref_image, alpha=0.7)
    plt.imshow(
        np.clip(class_sal_map, 0, 1),
        cmap='jet', alpha=0.3
    )
    plt.clim(0, 1)
    plt.colorbar(**colorbar_kwargs)
    plt.title(f"Class #{i+1} Pos Saliency")
    plt.axis('off')
    # Negative half saliency
    plt.subplot(2, sub_plot_ind, sub_plot_ind+2+i)
    plt.imshow(ref_image, alpha=0.7)
    plt.imshow(
        np.clip(class_sal_map, -1, 0),
        cmap='jet_r', alpha=0.3
    )
    plt.clim(-1, 0)
    plt.colorbar(**colorbar_kwargs)
    plt.title(f"Class #{i+1} Neg Saliency")
    plt.axis('off')
    plt.show()