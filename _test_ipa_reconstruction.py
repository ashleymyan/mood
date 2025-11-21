# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
from ipadapter_model import load_ipadapter
ip_model = load_ipadapter()
# %%
from PIL import Image
# Load two different images
basenames = ["04963", "02817"]
images = [
    Image.open(f"./experiments/TTL/left/{basename}.jpg").resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    for basename in basenames
]
images += [Image.open(f"./images/fingers.jpg").resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")]
# %%
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 14
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for ax in axes.flatten():
    ax.axis('off')

seeds = [1, 2, 3, 4]
for i, image in enumerate(images):
    # Show input image
    axes[i, 0].imshow(image)
    if i == 0:
        axes[i, 0].set_title(f"input")
    
    # Generate and show reconstructions
    for j in range(4):
        recon_image = ip_model.generate(pil_image=image, seed=seeds[j])
        axes[i, j+1].imshow(recon_image[0])
        if i == 0:
            axes[i, j+1].set_title(f"reconstructed\n seed {seeds[j]}")

fig.tight_layout()
plt.show()
# %%

