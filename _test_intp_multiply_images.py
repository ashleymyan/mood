#%%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_two_image_interpolation
from ipadapter_model import create_image_grid
# 
path1 = "./images/architecture_extra/A.jpg"
path2 = "./images/architecture_extra/B.jpg"
path3 = "./images/architecture_extra/C1.jpg"
path4 = "./images/architecture_extra/C2.jpg"
path5 = "./images/architecture_extra/C3.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image4 = Image.open(path4).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image5 = Image.open(path5).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
config_path = "./config.yaml"
single_image_model, _ = train_mood_space(
    pil_images=[image1, image2],
    learning_rate=0.001, 
    training_steps=1000,
    mlp_width=512,
    mlp_layers=4,
    config_path=config_path,
    n_eig=64,
)
multiple_image_model, _ = train_mood_space(
    pil_images=[image1, image2, image3, image4, image5],
    learning_rate=0.001, 
    training_steps=1000,
    mlp_width=512,
    mlp_layers=4,
    config_path=config_path,
    n_eig=64,
)
# %%
interpolation_weights = np.linspace(0, 1.0, 20).tolist()
fixed_correspondence = perform_two_image_interpolation(
    image1=image1,
    image2=image2,
    model=multiple_image_model,
    interpolation_weights=interpolation_weights,
    n_clusters=10,
    match_method='hungarian',
    use_dino_matching=True,
    use_two_step_clustering=False,
    n_subclusters_per_supercluster=6,
    config_path=config_path,
    return_matching=True,
)
single_interpolated_images = perform_two_image_interpolation(
    image1, 
    image2, 
    single_image_model,
    interpolation_weights,
    n_clusters=10, 
    match_method='hungarian',
    use_dino_matching=True,
    use_two_step_clustering=False,
    n_subclusters_per_supercluster=6,
    config_path=config_path,
    predefined_matching=fixed_correspondence,
)
multiple_interpolated_images = perform_two_image_interpolation(
    image1=image1,
    image2=image2,
    model=multiple_image_model,
    interpolation_weights=interpolation_weights,
    n_clusters=10,
    match_method='hungarian',
    use_dino_matching=True,
    use_two_step_clustering=False,
    n_subclusters_per_supercluster=6,
    config_path=config_path,
    return_matching=True,
    predefined_matching=fixed_correspondence,
)
# %%
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'monospace'
fig, axes = plt.subplots(3, 7, figsize=(10, 6))
for ax in axes.flatten():
    ax.axis('off')
# Top row: show the 5 images
axes[0, 1].imshow(image1)
axes[0, 2].imshow(image2)
axes[0, 3].imshow(image3)
axes[0, 4].imshow(image4)
axes[0, 5].imshow(image5)
axes[0, 1].set_title("image 1")
axes[0, 2].set_title("image 2")
axes[0, 3].set_title("extra image 1")
axes[0, 4].set_title("extra image 2")
axes[0, 5].set_title("extra image 3")
# Second row
axes[1, 0].imshow(image1)
axes[1, 6].imshow(image2)
axes[1, 0].set_title("input 1")
axes[1, 6].set_title("input 2")
axes[1, 1].imshow(single_interpolated_images[2])
title = f"no extra images"
axes[1, 0].text(-0.25, 0.5, title, fontsize=12, ha='left', va='center', rotation=90, transform=axes[1, 0].transAxes)
axes[1, 2].imshow(single_interpolated_images[5])
axes[1, 3].imshow(single_interpolated_images[10])
axes[1, 4].imshow(single_interpolated_images[15])
axes[1, 5].imshow(single_interpolated_images[18])
# Third row
axes[2, 0].imshow(image1)
axes[2, 6].imshow(image2)
axes[2, 0].set_title("input 1")
axes[2, 6].set_title("input 2")
axes[2, 1].imshow(multiple_interpolated_images[2])
title = f"with extra images"
axes[2, 0].text(-0.25, 0.5, title, fontsize=12, ha='left', va='center', rotation=90, transform=axes[2, 0].transAxes)
axes[2, 2].imshow(multiple_interpolated_images[5])
axes[2, 3].imshow(multiple_interpolated_images[10])
axes[2, 4].imshow(multiple_interpolated_images[15])
axes[2, 5].imshow(multiple_interpolated_images[18])
fig.tight_layout()
figure_dir = f"/workspace/experiments/architecture/figures_extra_images/"
import os
os.makedirs(figure_dir, exist_ok=True)
plt.savefig(os.path.join(figure_dir, "interpolation_extra_images.png"), bbox_inches='tight', dpi=144)
plt.close()
# plt.show()
# %%