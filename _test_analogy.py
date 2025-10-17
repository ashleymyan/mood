# %%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_three_image_analogy, perform_three_image_analogy_no_compression
from ipadapter_model import create_image_grid
import matplotlib.pyplot as plt
# 
path1 = "./images/jimi_portrait.jpg"
path2 = "./images/jimi_action.jpg"
path3 = "./images/bach_portrait.jpg"
path4 = "./images/violin.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image4 = Image.open(path4).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
grid = create_image_grid([image1, image2, image3], 1, 3)
grid
# %%
for i in range(10):
    config_path = "./config.yaml"
    model, trainer = train_mood_space(
        pil_images=[image1, image2, image3], 
        training_steps=1000,
        config_path=config_path,
        n_eig=64,
    )
    interpolation_weights = np.linspace(0.0, 2.0, 10).tolist()
    correspondence_plot, fig, interpolated_images = perform_three_image_analogy(
        image_list=[image3, image1, image2], 
        model=model, 
        interpolation_weights=interpolation_weights,
        n_clusters=10,
        match_method='hungarian',
    )
    all_images = interpolated_images

    display_size = (256, 256)
    resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
    result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
    plt.show()
    display(result_grid)


# %%
