# %%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_two_image_interpolation
from ipadapter_model import create_image_grid

path1 = "./images/jimi_portrait.jpg"
path2 = "./images/jimi_action.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")

model, trainer = train_mood_space(
    pil_images=[image1, image2], 
    learning_rate=0.001, 
    training_steps=1000,
    mlp_width=512,
    mlp_layers=4
)

interpolation_weights = np.linspace(0.0, 1.0, 10).tolist()
interpolated_images = perform_two_image_interpolation(
    image1=image1, 
    image2=image2, 
    model=model, 
    interpolation_weights=interpolation_weights,
    n_clusters=10, 
    match_method='hungarian',
    use_dino_matching=True
)
all_images = [image1] + interpolated_images + [image2]

display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
display(result_grid)

# %%



