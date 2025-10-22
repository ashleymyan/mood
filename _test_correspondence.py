#%%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_two_image_interpolation, get_correspondence_plot_from_two_images, get_correspondence_plot_from_multiple_images
from app import interpolate_two_images_no_compression
from ipadapter_model import create_image_grid
# 
path1 = "./images/playviolin.png"
path2 = "./images/playguitar.png"
# path1 = "./images/input_cat3.png"
# path2 = "./images/input_bread.png"
# path1 = "./images/duck1.jpg"
# path2 = "./images/toilet_paper.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
from extract_features import extract_dino_features, dino_image_transform
from dino_correspondence import ncut_tsne_multiple_images, kway_cluster_per_image, match_centers_two_images
images = torch.stack([dino_image_transform(image) for image in [image1, image2]])
dino_image_embeds = extract_dino_features(images)
n_clusters = 10
cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
a_to_b_mapping = match_centers_two_images(
    dino_image_embeds[0], dino_image_embeds[1],
    cluster_eigenvectors[0], cluster_eigenvectors[1], 
    match_method='hungarian'
)
# %%
a_cluster_labels = cluster_eigenvectors[0].argmax(-1)
b_cluster_labels = cluster_eigenvectors[1].argmax(-1)
mapped_b_cluster_labels = torch.zeros_like(b_cluster_labels)
for i, j in enumerate(a_to_b_mapping):
    mapped_b_cluster_labels[b_cluster_labels == j] = i
a_cluster_image = a_cluster_labels[1:].reshape(32, 32)
b_cluster_image = mapped_b_cluster_labels[1:].reshape(32, 32)
# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(a_cluster_image, cmap='tab20')
axes[0].set_title('A Cluster')
axes[1].imshow(b_cluster_image, cmap='tab20')
axes[1].set_title('B Cluster')
plt.suptitle(f'n_clusters={n_clusters}')
plt.tight_layout()
plt.show()
#%%