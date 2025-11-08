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
# path1 = "./images/input_pymaid.png"
# path2 = "./images/input_dom.png"
path1 = "./images/playviolin_hr.png"
path2 = "./images/playguitar_hr.png"
# path1 = "./images/coffee_cup.png"
# path2 = "./images/torus.png"
# path1 = "./images/rock1.png"
# path2 = "./images/rock2.png"
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
# %%
interpolation_weights = np.linspace(0.0, 1.0, 11).tolist()
interpolated_images1 = perform_two_image_interpolation(
    image1=image1, 
    image2=image2, 
    model=model, 
    interpolation_weights=interpolation_weights,
    n_clusters=10, 
    match_method='hungarian',
    use_multiscale_matching=False,
    use_dino_matching=True,
    config_path=config_path,
    predefined_matching=None,
)
for img in interpolated_images1:
    display(img)

# %%
# all_images = [image1] + interpolated_images + [image2]
all_images = interpolated_images1
display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 2, len(resized_images)//2)
result_grid
# %%

# %%
from app import interpolate_two_images_no_compression
interpolation_weights = np.linspace(0.5, 0.55, 3).tolist()
config_path = "./config.yaml"
interpolated_images2 = interpolate_two_images_no_compression(
    image1=image1, 
    image2=image2, 
    interpolation_weights=interpolation_weights,
    n_clusters=10,
    dino_matching=False,
    config_path=config_path,
    predefined_matching=None,
)

all_images = interpolated_images2
display_size = (256, 256)
resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
result_grid = create_image_grid(resized_images, 1, len(resized_images)//1)
result_grid
# %%
from rembg import remove
for image in interpolated_images2:
    image = remove(image)
    display(image)
# %%
import zipfile
from io import BytesIO

# Calculate the starting and ending index for the middle 6 images
start_index = len(resized_images) // 2 - 3
end_index = start_index + 6

# Select the middle 6 images
# middle_images = interpolated_images2[start_index:end_index]
middle_images = interpolated_images2
# Create a zip file in memory
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    for i, img in enumerate(middle_images):
        # Save each image to the zip file
        img_filename = f"image_{i+1}.png"
        with BytesIO() as img_bytes:
            img.save(img_bytes, format="PNG")
            zip_file.writestr(img_filename, img_bytes.getvalue())

# Save the zip file to disk
with open("baseline_cup_torus.zip", "wb") as f:
    f.write(zip_buffer.getvalue())

# %%
# save middle six images to disk
middle_images = interpolated_images1
for i, img in enumerate(middle_images):
    import os

    def get_available_filename(base_path, index):
        filename = f"{base_path}/intp{index}.png"
        while os.path.exists(filename):
            index += 1
            filename = f"{base_path}/intp{index}.png"
        return filename

    base_path = "images/cup_torus"
    filename = get_available_filename(base_path, i+1)
    img.save(filename)
# %%
interpolation_weights = np.linspace(0.3, 1.0, 10).tolist()
for _ in range(10):
    middle_images = interpolate_two_images_no_compression(
        image1=image1, 
        image2=image2, 
        interpolation_weights=interpolation_weights,
        n_clusters=10,
        dino_matching=True,
        config_path=config_path,
        predefined_matching=None,
    )
    for i, img in enumerate(middle_images):
        import os

        def get_available_filename(base_path, index):
            filename = f"{base_path}/baseline{index}.png"
            while os.path.exists(filename):
                index += 1
                filename = f"{base_path}/baseline{index}.png"
            return filename

        base_path = "images/cup_torus"
        filename = get_available_filename(base_path, i+1)
        img.save(filename)

# %%