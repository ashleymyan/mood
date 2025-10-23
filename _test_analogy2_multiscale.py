# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from PIL import Image
import numpy as np
from app import train_mood_space
from app import method2_analogy, method2_analogy_multi_corr
from ipadapter_model import create_image_grid
import matplotlib.pyplot as plt
# 
path1 = "./images/jimi_portrait.jpg"
path2 = "./images/jimi_action.jpg"
path3 = "./images/bach_portrait.jpg"
# path4 = "./images/violin.jpg"
# path1 = "./images/input_cat.png"
# path2 = "./images/input_bread.png"
# path3 = "./images/input_cat3.png"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# image4 = Image.open(path4).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
grid = create_image_grid([image1, image2, image3], 1, 3)
grid
# %%
config_path = "./config.yaml"
model, trainer = train_mood_space(
    pil_images=[image1, image2], 
    training_steps=1000,
    config_path=config_path,
)
# %%
interpolation_weights = np.linspace(0.0, 1.5, 11).tolist()
interpolated_images = method2_analogy_multi_corr(
    image_list=[image3, image1, image2], 
    model=model, 
    interpolation_weights=interpolation_weights,
    skip_a1a2_matching=True,
)
all_images = interpolated_images

display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
# result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
# display(result_grid)
# result_grid.save("./analogy.jpg")

fig, axes = plt.subplots(1, len(interpolated_images), figsize=(20, 2))
for i, img in enumerate(interpolated_images):
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"t={interpolation_weights[i]:.2f}")
fig.tight_layout()
plt.show()

# %%
