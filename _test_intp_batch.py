# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
from app import train_mood_space, perform_two_image_interpolation
from app import interpolate_two_images_no_compression
from ipadapter_model import create_image_grid
import numpy as np
import shutil

left_dir = "/workspace/experiments/TTL/left"
right_dir = "/workspace/experiments/TTL/right"
for left_image in sorted(os.listdir(left_dir))[::-1]:
    left_path = os.path.join(left_dir, left_image)
    right_path = os.path.join(right_dir, left_image.replace("left", "right"))
    image1 = Image.open(left_path).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    image2 = Image.open(right_path).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    input_image_prompt = ""
    basename = left_image.split(".")[0]
    
    output_dir = f"/workspace/experiments/TTL/results/VibeSpace/{basename}"
    if os.path.exists(output_dir):
        continue
    os.makedirs(output_dir, exist_ok=True)
    
    image1 = Image.open(left_path).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    image2 = Image.open(right_path).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    config_path = "./config.yaml"
    try:
        model, trainer = train_mood_space(
            pil_images=[image1, image2],
            learning_rate=0.001, 
            training_steps=1000,
            mlp_width=512,
            mlp_layers=4,
            config_path=config_path,
            n_eig=64,
        )
    except Exception as e:
        print(f"Error training model for {basename}: {e}")
        shutil.rmtree(output_dir, ignore_errors=True)
        continue
    
    done = False
    while not done:
        interpolation_weights = np.linspace(0., 1.0, 11).tolist()
        interpolated_images = perform_two_image_interpolation(
            image1, 
            image2, 
            model,
            interpolation_weights,
            n_clusters=20, 
            match_method='hungarian',
            use_dino_matching=True,
            use_two_step_clustering=False,
            config_path=config_path
        )
        is_black_image = False
        for img in interpolated_images:
            if np.all(np.array(img) == 0):
                is_black_image = True
                break
        if not is_black_image:
            done = True
    all_images = interpolated_images + [image2]
    # all_images = interpolated_images
    display_size = (512, 512)
    resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
    result_grid = create_image_grid(resized_images, 3, len(resized_images)//3)
    result_grid.save(f"{output_dir}/result_grid_{basename}.png")
    for i, img in enumerate(interpolated_images):
        img.save(f"{output_dir}/intp{i}.png")

# %%
