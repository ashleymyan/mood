#%%
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, perform_two_image_interpolation
from ipadapter_model import create_image_grid
# 
path1 = "./images/playviolin_hr.png"
path2 = "./images/playguitar_hr.png"
path3 = "./images/playlute_hr.png"
# path1 = "./experiments/my_images/left/restroom.png"
# path2 = "./experiments/my_images/right/restroom.png"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
config_path = "./config.yaml"
model, trainer = train_mood_space(
    pil_images=[image1, image2, image3],
    learning_rate=0.001, 
    training_steps=1000,
    mlp_width=512,
    mlp_layers=4,
    config_path=config_path,
    n_eig=64,
)
# %%
interpolation_weights = np.linspace(0, 1.0, 20).tolist()
interpolated_images = perform_two_image_interpolation(
    image1, 
    image2, 
    model,
    interpolation_weights,
    n_clusters=10, 
    match_method='hungarian',
    use_dino_matching=True,
    use_two_step_clustering=False,
    n_subclusters_per_supercluster=6,
    config_path=config_path
)
all_images = interpolated_images
# all_images = interpolated_images
display_size = (512, 512)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 4, len(resized_images)//4)
result_grid

# %%
import zipfile
import io

zip_filename = 'guitar_violin_blend_with_lute.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for idx, img in enumerate(interpolated_images):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        zipf.writestr(f"interpolated_{idx:02d}.png", img_byte_arr.read())
print(f"All interpolated images have been saved to {zip_filename}")
# %%
idxs = [7, 12, 14]
all_images = [image1] + [interpolated_images[i] for i in idxs] + [image2]
display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 1, len(resized_images)//1)
result_grid
# %%