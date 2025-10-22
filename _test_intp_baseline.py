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
config_path = "./config.yaml"
interpolation_weights = np.linspace(0., 1.0, 10).tolist()
interpolated_images = interpolate_two_images_no_compression(
    image1, 
    image2, 
    interpolation_weights,
    n_clusters=10, 
    match_method='hungarian',
    dino_matching=True,
    config_path=config_path
)
all_images = [image1] + interpolated_images + [image2]
# all_images = interpolated_images
display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
result_grid

# %%

