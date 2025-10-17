# %%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, interpolate_two_images_no_compression
from ipadapter_model import create_image_grid
import glob

path1 = "./images/jimi_portrait.jpg"
path2 = "./images/jimi_action.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")

config_path = "./config.yaml"
for i in range(3):

    interpolation_weights = np.linspace(0.0, 1.0, 10).tolist()
    interpolated_images = interpolate_two_images_no_compression(
        image1=image1, 
        image2=image2, 
        interpolation_weights=interpolation_weights,
        n_clusters=10, 
        match_method='hungarian',
        dino_matching=True,
        config_path=config_path
    )
    all_images = [image1] + interpolated_images + [image2]

    display_size = (256, 256)
    resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
    result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
    
    print("="*42)
    display(result_grid)
    print("="*42)
# %%



