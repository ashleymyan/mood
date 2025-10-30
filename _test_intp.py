# %%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_two_image_interpolation, get_correspondence_plot_from_two_images, get_correspondence_plot_from_multiple_images
from ipadapter_model import create_image_grid
# 
# path1 = "./images/jimi_portrait.jpg"
# path2 = "./images/jimi_action.jpg"
# path3 = "./images/bach_portrait.jpg"
# path4 = "./images/violin.jpg"
# path1 = "./images/dog1.jpg"
# path2 = "./images/fish.jpg"
# path1 = "./images/playviolin.png"
# path2 = "./images/playguitar.png"
# path1 = "./images/duck1.jpg"
# path2 = "./images/toilet_paper.jpg"
# path1 = "./images/input_cat.png"
# path2 = "./images/input_bread.png"
path1 = "./images/input_pymaid.png"
path2 = "./images/input_dom.png"
# path1 = "./images/guitar.png"
# path2 = "./images/violin.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# image4 = Image.open(path4).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
# correspondence_plot = get_correspondence_plot_from_multiple_images(
#     image_list=[image1, image2, image3],
#     n_clusters=10,
#     match_method='argmin'
# )
# %%
config_path = "./config.yaml"
model, trainer = train_mood_space(
    # pil_images=[image1, image2, image3], 
    pil_images=[image1, image2],
    learning_rate=0.001, 
    training_steps=1000,
    mlp_width=512,
    mlp_layers=4,
    config_path=config_path,
    n_eig=64,
)
#%%
# interpolation_weights = np.linspace(0.2, 0.8, 6).tolist()
# interpolation_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
interpolation_weights = np.linspace(0.0, 1.0, 10).tolist()
interpolated_images = perform_two_image_interpolation(
    image1=image1, 
    image2=image2, 
    model=model, 
    interpolation_weights=interpolation_weights,
    n_clusters=10, 
    match_method='hungarian',
    use_multiscale_matching=False,
    use_dino_matching=True,
    config_path=config_path
)
# all_images = [image1] + interpolated_images + [image2]
all_images = interpolated_images
display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
result_grid

# %%
