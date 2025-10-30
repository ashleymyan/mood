#%%
import torch
from PIL import Image
from ipadapter_model import create_image_grid
from dino_correspondence import _kway_cluster_single_image
# 
path1 = "./images/jimi_portrait.jpg"
path2 = "./images/jimi_action.jpg"
# path1 = "./images/input_cat3.png"
# path2 = "./images/input_bread.png"
# path1 = "./images/duck1.jpg"
# path2 = "./images/toilet_paper.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
create_image_grid([image1, image2], 1, 2)
# %%
from extract_features import extract_dino_features, dino_image_transform
from dino_correspondence import ncut_tsne_multiple_images, kway_cluster_per_image, match_centers_two_images
images = torch.stack([dino_image_transform(image) for image in [image1, image2]])
dino_image_embeds = extract_dino_features(images)
n_clusters1 = 6
n_clusters2 = 6
cluster_eigenvectors1 = _kway_cluster_single_image(dino_image_embeds[0], n_clusters=n_clusters1, gamma=None)
cluster_eigenvectors2 = _kway_cluster_single_image(dino_image_embeds[1], n_clusters=n_clusters2, gamma=None)
a_to_b_mapping = match_centers_two_images(
    dino_image_embeds[0], dino_image_embeds[1],
    cluster_eigenvectors1, cluster_eigenvectors2, 
    match_method='hungarian'
)
# %%
a_cluster_labels = cluster_eigenvectors1.argmax(-1)
b_cluster_labels = cluster_eigenvectors2.argmax(-1)
mapped_b_cluster_labels = torch.zeros_like(b_cluster_labels) / 0
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
plt.suptitle(f'n_clusters1={n_clusters1}, n_clusters2={n_clusters2}')
plt.tight_layout()
plt.show()
#%%