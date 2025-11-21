#%%
import torch
from PIL import Image
from ipadapter_model import create_image_grid
from dino_correspondence import _kway_cluster_single_image
# 
# path1 = "./images/stand_horse_man.png"
# path2 = "./images/ride_horse.png"
basename = "04963"
path1 = f"./experiments/TTL/left/{basename}.jpg"
path2 = f"./experiments/TTL/right/{basename}.jpg"
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
n_clusters1 = 20
n_clusters2 = 20
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
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 0.31, 1, 1], wspace=0.02, left=0, right=1, top=1, bottom=0)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 3])
ax3 = fig.add_subplot(gs[0, 4])
for ax in [ax0, ax1, ax2, ax3]:
    ax.axis('off')
ax0.imshow(image1)
ax1.imshow(a_cluster_image, cmap='tab20_r')
ax2.imshow(b_cluster_image, cmap='tab20_r')
ax3.imshow(image2)
plt.show()
#%%
from app import perform_two_image_interpolation, train_mood_space
config_path = "./config.yaml"
model, trainer = train_mood_space(
    pil_images=[image1, image2],
    learning_rate=0.001, 
    training_steps=1000,
    mlp_width=512,
    mlp_layers=4,
    config_path=config_path,
    n_eig=64,
)
# %%
import numpy as np
interpolation_weights = np.linspace(0.2, 1.0, 40).tolist()
interpolated_images = perform_two_image_interpolation(
    image1, 
    image2, 
    model,
    interpolation_weights,
    n_clusters=20, 
    match_method='hungarian',
    use_dino_matching=True,
    config_path=config_path,
    predefined_matching=([cluster_eigenvectors1, cluster_eigenvectors2], a_to_b_mapping)
)
all_images = interpolated_images
# all_images = interpolated_images
display_size = (512, 512)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 10, len(resized_images)//10)
result_grid
# %%
display(interpolated_images[13])
display(interpolated_images[19])
display(interpolated_images[25])
display(interpolated_images[28])
display(interpolated_images[32])
# %%