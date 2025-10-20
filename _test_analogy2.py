# %%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_three_image_analogy, perform_three_image_analogy_no_compression
from app import decoded_clip_space_analogy, perform_two_image_interpolation
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
# grid = create_image_grid([image1, image2, image3], 1, 3)
# grid
# %%
config_path = "./config.yaml"
model, trainer = train_mood_space(
    pil_images=[image1, image2], 
    training_steps=1000,
    config_path=config_path,
)
# %%
interpolation_weights = np.linspace(1.0, 3.0, 10).tolist()
interpolated_images = decoded_clip_space_analogy(
    image_list=[image3, image1, image2], 
    model=model, 
    interpolation_weights=interpolation_weights,
    n_clusters=100,
    n_clusters2=30,
    skip_a1a2_matching=True,
    use_a1a2_dino_matching=False,
    use_a1a2_pertoken_matching=True,
    match_method='hungarian',
)
all_images = interpolated_images

display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
result_grid.save("./analogy.jpg")
# %%
# interpolation_weights = np.linspace(0.0, 1.0, 10).tolist()
# interpolated_images = perform_two_image_interpolation(
#     image1=image1,
#     image2=image2,
#     model=model,
#     interpolation_weights=interpolation_weights,
#     n_clusters=100,
#     match_method='hungarian',
#     config_path=config_path,
# )
# all_images = [image1] + interpolated_images + [image2]

# display_size = (256, 256)
# resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
# result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
# display(result_grid)
# # %%

# %%
