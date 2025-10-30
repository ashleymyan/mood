"""
Mood Space Interactive Demo Application

This Gradio application provides an interactive interface for training and using
Mood Space compression models. The app includes three main functionalities:

1. Train a Mood Space compression model from uploaded images
2. Interpolate between two images using the trained model
3. Perform path lifting (analogy) given A1->B1, infer A2->B2

The application uses DINO/DINOv3 features for geometric understanding and
CLIP features for semantic representation, combined with neural compression
to learn a meaningful "mood space" representation.
"""

import copy
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

# Local imports
from compression_model import CompressionModel, train_compression_model, clear_gpu_memory
from dino_correspondence import (
    get_correspondence_plot, ncut_tsne_multiple_images, 
    kway_cluster_per_image, get_discrete_colors_from_clusters, 
    match_centers_three_images, match_centers_two_images, 
    get_cluster_center_features,
    hungarian_match_centers,
    kway_cluster_per_image_two_step, match_centers_two_step,
    kway_cluster_per_image_two_step_fgbg, match_centers_two_step_fgbg
)
from extract_features import (
    extract_clip_features,
    dino_image_transform, clip_image_transform, image_inverse_transform
)
from extract_features import extract_dino_features
# from extract_features import extract_dinov3_features as extract_dino_features
from dino_correspondence import _kway_cluster_single_image
from gradio_utils import add_download_button
from ipadapter_model import create_image_grid, generate_images_from_clip_embeddings
from ipadapter_model import load_ipadapter
from intrinsic_dim import estimate_intrinsic_dimension

# Configure matplotlib for consistent styling
plt.rcParams['font.family'] = 'monospace'

# Configuration
USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "true"
DEFAULT_IMAGES_PATH = ["./images/black_bear1.jpg", "./images/black_bear2.jpg", "./images/pink_bear1.jpg"]
DEFAULT_CONFIG_PATH = "./config.yaml"

# Optional HuggingFace Spaces integration
if USE_HUGGINGFACE_ZEROGPU:
    try:
        import spaces
    except ImportError:
        USE_HUGGINGFACE_ZEROGPU = False
        logging.warning("HuggingFace Spaces not available, running without GPU acceleration")


# ===== Utility Functions =====

def load_config(config_path):
    cfg_base = OmegaConf.load(DEFAULT_CONFIG_PATH)
    cfg = OmegaConf.load(config_path)
    cfg_base.update(cfg)
    return cfg_base

def load_default_images() -> List[Image.Image]:
    """Load default example images for the demo."""
    try:
        return [Image.open(image_path) for image_path in DEFAULT_IMAGES_PATH]
    except Exception as e:
        logging.warning(f"Could not load default images: {e}")
        return []


def load_gradio_images_helper(pil_images: Union[List, Image.Image, str]) -> List[Image.Image]:
    """
    Convert various image input formats to a list of PIL Images.
    """
    if pil_images is None:
        return []
    
    # Handle single image
    if isinstance(pil_images, Image.Image):
        return [pil_images.convert("RGB")]
    
    if isinstance(pil_images, str):
        return [Image.open(pil_images).convert("RGB")]
    
    # Handle list of images
    processed_images = []
    for image in pil_images:
        if isinstance(image, tuple):  # Gradio gallery format
            image = image[0]
        
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass  # Already PIL Image
        else:
            continue
        
        processed_images.append(image.convert("RGB"))
    
    return processed_images


# ===== Core Training Functions =====

def train_mood_space(pil_images: List[Image.Image], 
                    learning_rate: float = 0.001,
                    training_steps: int = 5000, 
                    mlp_width: int = 512,
                    mlp_layers: int = 4, 
                    n_eig: int = None,
                    mood_dimension: Optional[int] = None,
                    config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[CompressionModel, object]:
    """
    Train a Mood Space compression model from input images.
    
    This function extracts DINO and CLIP features from the input images,
    estimates the intrinsic dimensionality if not provided, and trains
    a neural compression model to learn a meaningful embedding space.
    
    Args:
        pil_images: List of PIL Images for training
    """
    # Load and configure training parameters
    config = load_config(config_path)
    
    # Process input images
    images = load_gradio_images_helper(pil_images)
    if len(images) == 0:
        raise ValueError("No valid images provided for training")
    
    # Transform images for feature extraction
    dino_input_images = torch.stack([dino_image_transform(image) for image in images])
    clip_input_images = torch.stack([clip_image_transform(image) for image in images])
    
    # Extract features using pre-trained models
    logging.info("Extracting DINO features...")
    dino_image_embeds = extract_dino_features(dino_input_images)
    
    logging.info("Extracting CLIP features...")
    clip_image_embeds = extract_clip_features(clip_input_images, ipadapter_version=config.ipadapter_version)
    
    # Determine target dimensionality
    if mood_dimension is None:
        flattened_features = dino_image_embeds.flatten(end_dim=-2)
        estimated_dim = estimate_intrinsic_dimension(flattened_features)
        mood_dimension = int(estimated_dim)
        logging.info(f"Estimated intrinsic dimension: {mood_dimension}")
    else:
        logging.info(f"Using user-specified dimension: {mood_dimension}")

    config.mood_dim = mood_dimension
    config.lr = learning_rate
    config.steps = training_steps
    config.latent_dim = mlp_width
    config.n_layer = mlp_layers
    if n_eig is not None:
        config.n_eig = n_eig
    
    # Create and train model
    model = CompressionModel(config, enable_gradio_progress=True)
    trainer = train_compression_model(
        model, config, dino_image_embeds, clip_image_embeds
    )
    
    return model, trainer


# Apply HuggingFace Spaces GPU decorator if available
if USE_HUGGINGFACE_ZEROGPU:
    train_mood_space = spaces.GPU(duration=60)(train_mood_space)


# ===== Direction Finding Functions =====

def compute_direction_from_three_images(image_embeds: torch.Tensor, 
                                       eigenvectors: torch.Tensor,
                                       a2_to_a1_mapping: np.ndarray, 
                                       a1_to_b1_mapping: np.ndarray) -> torch.Tensor:
    """
    Compute direction vectors for three-image analogy (A2, A1, B1).
    
    Given the correspondence mappings, this function computes the direction
    from A1 to B1 and applies it to A2 to predict B2.
    
    Args:
        image_embeds: Image embeddings [A2, A1, B1]
        eigenvectors: Cluster eigenvectors for each image
        a2_to_a1_mapping: Cluster mapping from A2 to A1
        a1_to_b1_mapping: Cluster mapping from A1 to B1
        
    Returns:
        torch.Tensor: Direction field for A2
    """
    n_clusters = eigenvectors[0].shape[-1]
    
    # Compute cluster centers for A1 and B1
    a1_center_features = get_cluster_center_features(
        image_embeds[1], eigenvectors[1].argmax(-1).cpu(), n_clusters
    )
    b1_center_features = get_cluster_center_features(
        image_embeds[2], eigenvectors[2].argmax(-1).cpu(), n_clusters
    )
    
    # Compute direction vectors from A1 to B1
    direction_vectors = []
    for i_a1, i_b1 in enumerate(a1_to_b1_mapping):
        direction = b1_center_features[i_b1] - a1_center_features[i_a1]
        direction_vectors.append(direction)
    direction_vectors = torch.stack(direction_vectors)
    
    # Apply direction to A2 based on cluster assignments
    cluster_labels = eigenvectors[0].argmax(-1).cpu()
    direction_field = torch.zeros_like(image_embeds[0])
    
    for i_cluster in range(n_clusters):
        cluster_mask = cluster_labels == i_cluster
        if cluster_mask.sum() > 0:
            mapped_direction = direction_vectors[a2_to_a1_mapping[i_cluster]]
            direction_field[cluster_mask] = mapped_direction
    
    return direction_field


def compute_direction_from_two_images(image_embeds: torch.Tensor, 
                                    eigenvectors: torch.Tensor | List[torch.Tensor],
                                    a_to_b_mapping: np.ndarray, 
                                    use_unit_norm: bool = False,
                                    return_direction_vectors: bool = False) -> torch.Tensor:
    """
    Compute direction vectors for two-image interpolation.
    
    Args:
        image_embeds: Image embeddings [A, B]
        eigenvectors: Cluster eigenvectors [A, B]
        a_to_b_mapping: Cluster mapping from A to B
        use_unit_norm: Whether to normalize direction vectors
        
    Returns:
        torch.Tensor: Direction field for image A
    """
    
    # Compute cluster centers
    a_center_features = get_cluster_center_features(
        image_embeds[0], eigenvectors[0].argmax(-1).cpu(), eigenvectors[0].shape[-1]
    )
    b_center_features = get_cluster_center_features(
        image_embeds[1], eigenvectors[1].argmax(-1).cpu(), eigenvectors[1].shape[-1]
    )
    
    # Compute direction vectors
    direction_vectors = []
    for i_a, i_b in enumerate(a_to_b_mapping):
        direction = b_center_features[i_b] - a_center_features[i_a]
        if use_unit_norm:
            direction = F.normalize(direction, dim=-1)
        direction_vectors.append(direction)
    direction_vectors = torch.stack(direction_vectors)
    
    if return_direction_vectors:
        return direction_vectors
    
    # Apply direction based on cluster assignments
    cluster_labels = eigenvectors[0].argmax(-1).cpu()
    direction_field = torch.zeros_like(image_embeds[0])
    
    for i_cluster in range(eigenvectors[0].shape[-1]):
        cluster_mask = cluster_labels == i_cluster
        if cluster_mask.sum() > 0:
            direction_field[cluster_mask] = direction_vectors[i_cluster]
    
    return direction_field


def compute_direction_from_two_images_two_step(
    image_embeds: torch.Tensor,
    subcluster_eigenvectors: torch.Tensor | List[torch.Tensor],
    subcluster_mapping: np.ndarray,
    use_unit_norm: bool = False,
    return_direction_vectors: bool = False
) -> torch.Tensor:
    """
    Compute direction vectors for two-image interpolation using 2-step clustering.
    
    Args:
        image_embeds: Image embeddings [A, B]
        subcluster_eigenvectors: Subcluster eigenvectors [A, B]
        subcluster_mapping: Subcluster mapping from A to B
        use_unit_norm: Whether to normalize direction vectors
        return_direction_vectors: If True, return direction vectors instead of field
        
    Returns:
        torch.Tensor: Direction field for image A (or direction vectors if return_direction_vectors=True)
    """
    # This function is similar to compute_direction_from_two_images but operates on subclusters
    return compute_direction_from_two_images(
        image_embeds, 
        subcluster_eigenvectors, 
        subcluster_mapping, 
        use_unit_norm, 
        return_direction_vectors
    )


def multiscale_directions(
    dino_embeds: torch.Tensor,
    compressed_embeds: torch.Tensor,
    n_cluster_list: List[int] = [10, 30],
    k2_expansion_factor: float = 1.,
    n_repeats: int = 1,  # TODO: fix this
):

    direction_fields = []
    for n_clusters in n_cluster_list:
        for i_repeat in range(n_repeats):
            dino_eigvec0 = _kway_cluster_single_image(dino_embeds[0], n_clusters=n_clusters, gamma=None)
            dino_eigvec1 = _kway_cluster_single_image(dino_embeds[1], n_clusters=int(n_clusters * k2_expansion_factor), gamma=None)
            a_to_b_mapping = match_centers_two_images(
                dino_embeds[0], dino_embeds[1], dino_eigvec0, dino_eigvec1, match_method='hungarian'
            )
            direction_field = compute_direction_from_two_images(
                compressed_embeds, [dino_eigvec0, dino_eigvec1], a_to_b_mapping, use_unit_norm=False
            )
            direction_fields.append(direction_field)
    direction_fields = torch.stack(direction_fields)
    direction_fields = direction_fields.mean(0)
    return direction_fields


def multiscale_directions_remap(
    dino_embeds: torch.Tensor,  # [A2, A1], [2, 197, 768]
    direction_field: torch.Tensor,  # [t, 197, 1280]
    n_cluster_list: List[int] = [30],
    k2_expansion_factor: float = 1.0,
    n_repeats: int = 10,
):

    new_direction_fields = []
    for n_clusters in n_cluster_list:
        for i_repeat in range(n_repeats):
            dino_eigvec0 = _kway_cluster_single_image(dino_embeds[0], n_clusters=n_clusters, gamma=None)
            dino_eigvec1 = _kway_cluster_single_image(dino_embeds[1], n_clusters=int(n_clusters * k2_expansion_factor), gamma=None)
            a2_to_a1_mapping = match_centers_two_images(
                dino_embeds[0], dino_embeds[1], dino_eigvec0, dino_eigvec1, match_method='hungarian'
            )
            new_direction_field = torch.zeros_like(direction_field)
            for i_a2, i_a1 in enumerate(a2_to_a1_mapping):
                i_a2_mask = dino_eigvec0.argmax(-1) == i_a2
                i_a1_mask = dino_eigvec1.argmax(-1) == i_a1
                if i_a1_mask.sum() > 0 and i_a2_mask.sum() > 0:
                    for i_t in range(direction_field.shape[0]):
                        new_direction_field[i_t, i_a2_mask] = direction_field[i_t, i_a1_mask].mean(dim=0)
            new_direction_fields.append(new_direction_field)
    new_direction_fields = torch.stack(new_direction_fields)
    new_direction_fields = new_direction_fields.mean(0)
    return new_direction_fields


# ===== Main Application Functions =====

def method2_analogy(image_list: List[Image.Image], 
                                model: CompressionModel,
                                interpolation_weights: List[float], 
                                n_clusters: int = 100,
                                n_clusters2: int = 30,
                                skip_a1a2_matching: bool = False,
                                use_a1a2_dino_matching: bool = True,
                                n_samples: int = 1, 
                                match_method: str = 'hungarian',
                                config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[Image.Image, plt.Figure, List[Image.Image]]:
    # images: [A2, A1, B1]
    clear_gpu_memory()
    config = load_config(config_path)
    images = torch.stack([dino_image_transform(image) for image in image_list[1:]])  # [A1, B1]
    dino_image_embeds = extract_dino_features(images)  # [A1, B1]
    
    dino_eigvec = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None) # dino matching for A1 and B1
    
    # blend from A1 to B1
    a1_to_b1_mapping = match_centers_two_images(dino_image_embeds[0], dino_image_embeds[1], dino_eigvec[0], dino_eigvec[1], match_method=match_method)
    
    m_embeds = model.encoder(dino_image_embeds)
    decode_clip_embeds = model.decoder(m_embeds)
    
    direction_field = compute_direction_from_two_images(
        m_embeds, dino_eigvec, a1_to_b1_mapping, False
    )  # A1 -> B1
    
    a1_directions = []
    for weight in interpolation_weights:
        interpolated_embedding = m_embeds[0] + direction_field * weight
        decompressed = model.decoder(interpolated_embedding)
        direction = decompressed - decode_clip_embeds[0]  # from A1
        a1_directions.append(direction)
    a1_directions = torch.stack(a1_directions)  # saved for lifting to A2
    
    
    # lift direction field (A1 -> B1) to A2
    if use_a1a2_dino_matching:
        dino_images = torch.stack([dino_image_transform(image) for image in image_list[:2]])
        dino_images = torch.nn.functional.interpolate(dino_images, size=(256, 256), mode='bilinear', align_corners=False)
        dino_image_embeds = extract_dino_features(dino_images)  # [A2, A1]
        dino_eigvec = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters2, gamma=None)
        a2_to_a1_mapping = match_centers_two_images(dino_image_embeds[0], dino_image_embeds[1], dino_eigvec[0], dino_eigvec[1], match_method=match_method)
        a2_directions = []
        for a1_direction in a1_directions:
            # upsample a1_direction from (1+16*16) to (1+32*32)
            def resample_direction(direction, size_from, size_to, mode='nearest'):
                cls_direction = direction[0].unsqueeze(0)
                patch_direction = direction[1:]
                patch_direction = patch_direction.view(size_from, size_from, -1)
                patch_direction = rearrange(patch_direction, 'h w c -> c h w').unsqueeze(0)
                patch_direction = torch.nn.functional.interpolate(patch_direction, size=(size_to, size_to), mode=mode)
                patch_direction = rearrange(patch_direction, 'b c h w -> b (h w) c').squeeze(0)
                direction = torch.cat([cls_direction, patch_direction], dim=0)
                return direction
            # a1_direction = resample_direction(a1_direction, 16, 32, mode='nearest')
            # direction is defined in clip space, on A1
            a2_direction = torch.zeros_like(a1_direction)
            for i_a2, i_a1 in enumerate(a2_to_a1_mapping):
                i_a2_mask = dino_eigvec[0].argmax(-1) == i_a2
                i_a1_mask = dino_eigvec[1].argmax(-1) == i_a1
                if i_a1_mask.sum() > 0:
                    a2_direction[i_a2_mask] = a1_direction[i_a1_mask].mean(dim=0)
            # downsample a2_direction from (1+32*32) to (1+16*16)
            # a2_direction = resample_direction(a2_direction, 32, 16, mode='bilinear')
            a2_directions.append(a2_direction)
            print(a2_direction.norm(1).mean())
        a2_directions = torch.stack(a2_directions)
    
    # if use_a1a2_pertoken_matching:
    #     dino_images = torch.stack([dino_image_transform(image) for image in image_list[:2]])
    #     dino_images = torch.nn.functional.interpolate(dino_images, size=(256, 256), mode='bilinear', align_corners=False)
    #     dino_image_embeds = extract_dino_features(dino_images)  # [A2, A1]
    #     dino_eigvec, _ = ncut_tsne_multiple_images(dino_image_embeds, n_eig=20)
    #     # a2_to_a1_mapping = hungarian_match_centers(dino_eigvec[0, 1:], dino_eigvec[1, 1:]) + 1
    #     a2_to_a1_mapping = hungarian_match_centers(dino_image_embeds[0, 1:], dino_image_embeds[1, 1:]) + 1
    #     a2_to_a1_mapping = np.concatenate([np.array([0]), a2_to_a1_mapping])
    #     a2_directions = torch.index_select(a1_directions, 1, torch.tensor(a2_to_a1_mapping))
        
        
    
    if skip_a1a2_matching:
        a2_directions = a1_directions
    
    a2_clip = extract_clip_features(clip_image_transform(image_list[0]).unsqueeze(0), 
                                    ipadapter_version=config.ipadapter_version)
    a2_clip = a2_clip.squeeze(0)
    # generate images from interpolated A2 clip embeddings
    ip_model = load_ipadapter(version=config.ipadapter_version)
    
    generated_images = []
    for direction in a2_directions:
        gen_images= generate_images_from_clip_embeddings(
            ip_model, a2_clip + direction, num_samples=n_samples
        )
        generated_images.extend(gen_images)
    
    del ip_model
    clear_gpu_memory()
    
    return generated_images


def method2_analogy_multi_corr(image_list: List[Image.Image], 
                                                model: CompressionModel,
                                                interpolation_weights: List[float], 
                                                skip_a1a2_matching: bool = False,
                                                n_samples: int = 1, 
                                                config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[Image.Image, plt.Figure, List[Image.Image]]:
    # images: [A2, A1, B1]
    clear_gpu_memory()
    config = load_config(config_path)
    images = torch.stack([dino_image_transform(image) for image in image_list[1:]])  # [A1, B1]
    dino_image_embeds = extract_dino_features(images)  # [A1, B1]
    m_embeds = model.encoder(dino_image_embeds)
    decode_clip_embeds = model.decoder(m_embeds)
    
    direction_field = multiscale_directions(dino_image_embeds, m_embeds, n_cluster_list=[10, 30], n_repeats=1)  # TODO: fix this  
    
    a1_directions = []
    for weight in interpolation_weights:
        interpolated_embedding = m_embeds[0] + direction_field * weight
        decompressed = model.decoder(interpolated_embedding)
        direction = decompressed - decode_clip_embeds[0]  # from A1
        a1_directions.append(direction)
    a1_directions = torch.stack(a1_directions)  # saved for lifting to A2
    
    dino_images = torch.stack([dino_image_transform(image) for image in image_list[:2]])
    dino_images = torch.nn.functional.interpolate(dino_images, size=(256, 256), mode='bilinear', align_corners=False)
    dino_image_embeds = extract_dino_features(dino_images)  # [A2, A1]
    
    a2_directions = multiscale_directions_remap(dino_image_embeds, a1_directions, n_cluster_list=[30], n_repeats=10)  
    
    if skip_a1a2_matching:
        a2_directions = a1_directions
    
    a2_clip = extract_clip_features(clip_image_transform(image_list[0]).unsqueeze(0), 
                                    ipadapter_version=config.ipadapter_version)
    a2_clip = a2_clip.squeeze(0)
    # generate images from interpolated A2 clip embeddings
    ip_model = load_ipadapter(version=config.ipadapter_version)
    
    generated_images = []
    for direction in a2_directions:
        gen_images= generate_images_from_clip_embeddings(
            ip_model, a2_clip + direction, num_samples=n_samples
        )
        generated_images.extend(gen_images)
    
    del ip_model
    clear_gpu_memory()
    
    return generated_images

    
def perform_three_image_analogy(image_list: List[Image.Image], 
                               model: CompressionModel,
                               interpolation_weights: List[float], 
                               n_clusters: int = 30,
                               n_samples: int = 1, 
                               match_method: str = 'hungarian',
                               config_path: str = DEFAULT_CONFIG_PATH,
                               do_cycle_consistency: bool = False) -> Union[Tuple[Image.Image, plt.Figure, List[Image.Image]], Tuple[Image.Image, plt.Figure, List[Image.Image], plt.Figure, List[Image.Image]]]:
    """
    Perform three-image analogy: given A2, A1, B1, predict A2 -> B2.
    
    Args:
        image_list: List of three images [A2, A1, B1]
        model: Trained compression model
        interpolation_weights: Interpolation weights for generation
        n_clusters: Number of clusters for correspondence matching
        n_samples: Number of samples to generate per weight
        match_method: Method for cluster matching ('hungarian' or 'argmin')
        
    Returns:
        Tuple of (correspondence_plot, interpolation_plot, generated_images)
    """
    clear_gpu_memory()
    config = load_config(config_path)
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(image) for image in image_list])
    dino_image_embeds = extract_dino_features(images)
    compressed_image_embeds = model.encoder(dino_image_embeds)
    
    # Compute correspondences and clustering
    joint_eigenvectors, joint_colors = ncut_tsne_multiple_images(dino_image_embeds, n_eig=30, gamma=None)
    cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
    discrete_colors = get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors)
    
    # Find cluster correspondences
    a2_to_a1_mapping, a1_to_b1_mapping = match_centers_three_images(
        dino_image_embeds, cluster_eigenvectors, match_method=match_method
    )
    
    # Compute direction field
    direction_field = compute_direction_from_three_images(
        compressed_image_embeds, cluster_eigenvectors, a2_to_a1_mapping, a1_to_b1_mapping
    )
    
    # Create correspondence visualization
    cluster_orders = [
        np.arange(n_clusters),
        a2_to_a1_mapping,
        a1_to_b1_mapping[a2_to_a1_mapping],
    ]
    correspondence_plot = get_correspondence_plot(
        images, cluster_eigenvectors, cluster_orders, discrete_colors, hw=16 * 2, n_cols=10
    )
    
    # Generate interpolated images
    ip_model = load_ipadapter(version=config.ipadapter_version)
    n_steps = len(interpolation_weights)
    generated_images = []
    
    fig, axes = plt.subplots(n_samples, n_steps, figsize=(n_steps * 2, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    progress = gr.Progress() if 'gr' in globals() else None
    
    for i_w, weight in enumerate(interpolation_weights):
        if progress:
            progress(i_w / n_steps, desc=f"Interpolating w={weight:.2f}")
        
        # Interpolate in compressed space
        interpolated_embedding = compressed_image_embeds[0] + direction_field * weight
        decompressed_embedding = model.decoder(interpolated_embedding)
        
        # Generate images from interpolated embeddings
        batch_images = generate_images_from_clip_embeddings(
            ip_model, decompressed_embedding, num_samples=n_samples  
        )
        generated_images.extend(batch_images)
        
        # Add to plot
        for i_sample in range(n_samples):
            ax = axes[i_sample, i_w]
            ax.imshow(batch_images[i_sample])
            ax.axis('off')
            if i_sample == 0:
                ax.set_title(f"w={weight:.2f}")
    
    fig.tight_layout()

    if do_cycle_consistency:
        # Currently we have the directions from A1 -> B1 and we are applying it to A2 to get B2
        # We can also compute the direction from A1 -> A2 and apply it to B1 to get B2
        n_clusters = cluster_eigenvectors[0].shape[-1]

        # Invert mappings to get A1->A2 and B1->A1
        a1_to_a2_mapping = np.full(n_clusters, -1, dtype=np.int64)
        for i_a2, i_a1 in enumerate(a2_to_a1_mapping):
            a1_to_a2_mapping[i_a1] = i_a2

        b1_to_a1_mapping = np.full(n_clusters, -1, dtype=np.int64)
        for i_a1, i_b1 in enumerate(a1_to_b1_mapping):
            b1_to_a1_mapping[i_b1] = i_a1

        # Compute cluster center features on compressed embeddings
        a1_centers = get_cluster_center_features(compressed_image_embeds[1], cluster_eigenvectors[1].argmax(-1).cpu(), n_clusters)
        a2_centers = get_cluster_center_features(compressed_image_embeds[0], cluster_eigenvectors[0].argmax(-1).cpu(), n_clusters)

        # Per-cluster direction A1 -> A2 in compressed space
        dir_a1_to_a2 = []
        for i_a1 in range(n_clusters):
            i_a2 = a1_to_a2_mapping[i_a1]
            if i_a2 < 0:
                dir_a1_to_a2.append(torch.zeros_like(a1_centers[i_a1]))
            else:
                dir_a1_to_a2.append(a2_centers[i_a2] - a1_centers[i_a1])
        dir_a1_to_a2 = torch.stack(dir_a1_to_a2)

        # Build direction field for B1 using B1->A1 then A1->A2 (compressed space)
        b1_labels = cluster_eigenvectors[2].argmax(-1).cpu()
        direction_for_b1 = torch.zeros_like(compressed_image_embeds[2])
        for j_b1 in range(n_clusters):
            mask = b1_labels == j_b1
            if mask.sum() > 0:
                i_a1 = b1_to_a1_mapping[j_b1]
                if i_a1 >= 0:
                    direction_for_b1[mask] = dir_a1_to_a2[i_a1]

        # Interpolate from B1 toward B2 along the constructed direction and plot
        fig_cycle, axes_cycle = plt.subplots(n_samples, n_steps, figsize=(n_steps * 2, n_samples * 3))
        if n_samples == 1:
            axes_cycle = axes_cycle.reshape(1, -1)

        cycle_generated_images = []
        for i_w, weight in enumerate(interpolation_weights):
            b1_interpolated = compressed_image_embeds[2] + direction_for_b1 * weight
            b1_decompressed = model.decoder(b1_interpolated)
            gen_images_b1 = generate_images_from_clip_embeddings(ip_model, b1_decompressed, num_samples=n_samples)
            cycle_generated_images.extend(gen_images_b1)
            for i_sample in range(n_samples):
                ax = axes_cycle[i_sample, i_w]
                ax.imshow(gen_images_b1[i_sample])
                ax.axis('off')
                if i_sample == 0:
                    ax.set_title(f"w={weight:.2f}")
        fig_cycle.tight_layout()
        
        del ip_model
        clear_gpu_memory()
        return correspondence_plot, fig, generated_images, fig_cycle, cycle_generated_images
        
    
    # Clean up
    del ip_model
    clear_gpu_memory()
    
    return correspondence_plot, fig, generated_images


def method2_analogy_multi_corr_no_compression(image_list: List[Image.Image], 
                                                interpolation_weights: List[float], 
                                                skip_a1a2_matching: bool = False,
                                                n_samples: int = 1, 
                                                config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[Image.Image, plt.Figure, List[Image.Image]]:
    # images: [A2, A1, B1]
    clear_gpu_memory()
    config = load_config(config_path)
    images = torch.stack([dino_image_transform(image) for image in image_list[1:]])  # [A1, B1]
    images = torch.nn.functional.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
    dino_image_embeds = extract_dino_features(images)  # [A1, B1]
    images = torch.stack([clip_image_transform(image) for image in image_list[1:]])  # [A1, B1]
    clip_image_embeds = extract_clip_features(images, ipadapter_version=config.ipadapter_version)
    
    direction_field = multiscale_directions(dino_image_embeds, clip_image_embeds, n_cluster_list=[10, 30], n_repeats=1)  # TODO: fix this  
    
    a1_directions = []
    for weight in interpolation_weights:
        direction = direction_field * weight
        a1_directions.append(direction)
    a1_directions = torch.stack(a1_directions)  # saved for lifting to A2
    
    dino_images = torch.stack([dino_image_transform(image) for image in image_list[:2]])  # [A2, A1]
    dino_images = torch.nn.functional.interpolate(dino_images, size=(256, 256), mode='bilinear', align_corners=False)
    dino_image_embeds = extract_dino_features(dino_images)  # [A2, A1]
    
    a2_directions = multiscale_directions_remap(dino_image_embeds, a1_directions, n_cluster_list=[30], n_repeats=10)  
    
    if skip_a1a2_matching:
        a2_directions = a1_directions
    
    a2_clip = extract_clip_features(clip_image_transform(image_list[0]).unsqueeze(0), 
                                    ipadapter_version=config.ipadapter_version)
    a2_clip = a2_clip.squeeze(0)
    # generate images from interpolated A2 clip embeddings
    ip_model = load_ipadapter(version=config.ipadapter_version)
    
    generated_images = []
    for direction in a2_directions:
        gen_images= generate_images_from_clip_embeddings(
            ip_model, a2_clip + direction, num_samples=n_samples
        )
        generated_images.extend(gen_images)
    
    del ip_model
    clear_gpu_memory()
    
    return generated_images


def perform_three_image_analogy_no_compression(image_list: List[Image.Image], 
                               interpolation_weights: List[float], 
                               n_clusters: int = 30,
                               n_samples: int = 1, 
                               match_method: str = 'hungarian',
                               config_path: str = DEFAULT_CONFIG_PATH,
                               do_cycle_consistency: bool = False) -> Union[Tuple[Image.Image, plt.Figure, List[Image.Image]], Tuple[Image.Image, plt.Figure, List[Image.Image], plt.Figure, List[Image.Image]]]:
    """
    Perform three-image analogy: given A2, A1, B1, predict A2 -> B2.
    
    Args:
        image_list: List of three images [A2, A1, B1]
        model: Trained compression model
        interpolation_weights: Interpolation weights for generation
        n_clusters: Number of clusters for correspondence matching
        n_samples: Number of samples to generate per weight
        match_method: Method for cluster matching ('hungarian' or 'argmin')
        
    Returns:
        Tuple of (correspondence_plot, interpolation_plot, generated_images)
    """
    clear_gpu_memory()
    config = load_config(config_path)
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(image) for image in image_list])
    # downsample to 256x256
    images = torch.nn.functional.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
    dino_image_embeds = extract_dino_features(images)
    images = torch.stack([clip_image_transform(image) for image in image_list])
    clip_image_embeds = extract_clip_features(images, ipadapter_version=config.ipadapter_version)
    
    # Compute correspondences and clustering
    joint_eigenvectors, joint_colors = ncut_tsne_multiple_images(dino_image_embeds, n_eig=30, gamma=None)
    cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
    discrete_colors = get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors)
    
    # Find cluster correspondences
    a2_to_a1_mapping, a1_to_b1_mapping = match_centers_three_images(
        dino_image_embeds, cluster_eigenvectors, match_method=match_method
    )
    
    # Compute direction field
    direction_field = compute_direction_from_three_images(
        clip_image_embeds, cluster_eigenvectors, a2_to_a1_mapping, a1_to_b1_mapping
    )
    
    # Create correspondence visualization
    cluster_orders = [
        np.arange(n_clusters),
        a2_to_a1_mapping,
        a1_to_b1_mapping[a2_to_a1_mapping],
    ]
    correspondence_plot = None
    
    # Generate interpolated images
    ip_model = load_ipadapter(version=config.ipadapter_version)
    n_steps = len(interpolation_weights)
    generated_images = []
    
    fig, axes = plt.subplots(n_samples, n_steps, figsize=(n_steps * 2, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    progress = gr.Progress() if 'gr' in globals() else None
    
    for i_w, weight in enumerate(interpolation_weights):
        if progress:
            progress(i_w / n_steps, desc=f"Interpolating w={weight:.2f}")
        
        # Interpolate in compressed space
        interpolated_embedding = clip_image_embeds[0] + direction_field * weight
        
        # Generate images from interpolated embeddings
        batch_images = generate_images_from_clip_embeddings(
            ip_model, interpolated_embedding, num_samples=n_samples  
        )
        generated_images.extend(batch_images)
        
        # Add to plot
        for i_sample in range(n_samples):
            ax = axes[i_sample, i_w]
            ax.imshow(batch_images[i_sample])
            ax.axis('off')
            if i_sample == 0:
                ax.set_title(f"w={weight:.2f}")
    
    fig.tight_layout()

    if do_cycle_consistency:
        # Currently we have the directions from A1 -> B1 and we are applying it to A2 to get B2
        # We can also compute the direction from A1 -> A2 and apply it to B1 to get B2
        n_clusters = cluster_eigenvectors[0].shape[-1]

        # Invert mappings to get A1->A2 and B1->A1
        a1_to_a2_mapping_inv = np.full(n_clusters, -1, dtype=np.int64)
        for i_a2, i_a1 in enumerate(a2_to_a1_mapping):
            a1_to_a2_mapping_inv[i_a1] = i_a2

        b1_to_a1_mapping_inv = np.full(n_clusters, -1, dtype=np.int64)
        for i_a1, i_b1 in enumerate(a1_to_b1_mapping):
            b1_to_a1_mapping_inv[i_b1] = i_a1

        # Compute cluster center features on clip embeddings
        a1_centers = get_cluster_center_features(clip_image_embeds[1], cluster_eigenvectors[1].argmax(-1).cpu(), n_clusters)
        a2_centers = get_cluster_center_features(clip_image_embeds[0], cluster_eigenvectors[0].argmax(-1).cpu(), n_clusters)

        # Per-cluster direction A1 -> A2 in clip space
        dir_a1_to_a2 = []
        for i_a1 in range(n_clusters):
            i_a2 = a1_to_a2_mapping_inv[i_a1]
            if i_a2 < 0:
                dir_a1_to_a2.append(torch.zeros_like(a1_centers[i_a1]))
            else:
                dir_a1_to_a2.append(a2_centers[i_a2] - a1_centers[i_a1])
        dir_a1_to_a2 = torch.stack(dir_a1_to_a2)

        # Build direction field for B1 using B1->A1 then A1->A2 (clip space)
        b1_labels = cluster_eigenvectors[2].argmax(-1).cpu()
        direction_for_b1 = torch.zeros_like(clip_image_embeds[2])
        for j_b1 in range(n_clusters):
            mask = b1_labels == j_b1
            if mask.sum() > 0:
                i_a1 = b1_to_a1_mapping_inv[j_b1]
                if i_a1 >= 0:
                    direction_for_b1[mask] = dir_a1_to_a2[i_a1]

        # Interpolate from B1 toward B2 along the constructed direction and plot
        fig_cycle, axes_cycle = plt.subplots(n_samples, n_steps, figsize=(n_steps * 2, n_samples * 3))
        if n_samples == 1:
            axes_cycle = axes_cycle.reshape(1, -1)

        cycle_generated_images = []
        for i_w, weight in enumerate(interpolation_weights):
            b1_interpolated = clip_image_embeds[2] + direction_for_b1 * weight
            gen_images_b1 = generate_images_from_clip_embeddings(ip_model, b1_interpolated, num_samples=n_samples)
            cycle_generated_images.extend(gen_images_b1)
            for i_sample in range(n_samples):
                ax = axes_cycle[i_sample, i_w]
                ax.imshow(gen_images_b1[i_sample])
                ax.axis('off')
                if i_sample == 0:
                    ax.set_title(f"w={weight:.2f}")
        fig_cycle.tight_layout()
        
        del ip_model
        clear_gpu_memory()
        return correspondence_plot, fig, generated_images, fig_cycle, cycle_generated_images
    
    # Clean up
    del ip_model
    clear_gpu_memory()
    
    return correspondence_plot, fig, generated_images


def get_correspondence_plot_from_two_images(image1: Image.Image, image2: Image.Image, 
                               n_clusters: int = 30,
                               match_method: str = 'hungarian') -> Image.Image:
    """
    Get the correspondence plot for three images.
    
    Args:
        image_list: List of three images [A2, A1, B1]
        n_clusters: Number of clusters for correspondence matching
        match_method: Method for cluster matching ('hungarian' or 'argmin')
    Returns:
        correspondence_plot: Correspondence plot
    """
    clear_gpu_memory()
    
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(image) for image in [image1, image2]])
    dino_image_embeds = extract_dino_features(images)
    
    # Compute correspondences and clustering
    joint_eigenvectors, joint_colors = ncut_tsne_multiple_images(dino_image_embeds, n_eig=30, gamma=None)
    cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
    discrete_colors = get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors)
    
    # Find cluster correspondences
    a_to_b_mapping = match_centers_two_images(
        dino_image_embeds[0], dino_image_embeds[1], cluster_eigenvectors[0], cluster_eigenvectors[1], match_method=match_method
    )
    
    # Create correspondence visualization
    cluster_orders = [
        np.arange(n_clusters),
        a_to_b_mapping
    ]
    correspondence_plot = get_correspondence_plot(
        images, cluster_eigenvectors, cluster_orders, discrete_colors, hw=16 * 2, n_cols=10
    )
    return correspondence_plot
    

def get_correspondence_plot_from_multiple_images(image_list: List[Image.Image], 
                                                n_clusters: int = 30,
                                                match_method: str = 'hungarian') -> Image.Image:
    clear_gpu_memory()
    
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(image) for image in image_list])
    dino_image_embeds = extract_dino_features(images)
    
    # Compute correspondences and clustering
    joint_eigenvectors, joint_colors = ncut_tsne_multiple_images(dino_image_embeds, n_eig=30, gamma=None)
    cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
    discrete_colors = get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors)
    
    cluster_orders = [
    ]
    for i in range(len(image_list)):
        matching = match_centers_two_images(
            dino_image_embeds[0], dino_image_embeds[i], cluster_eigenvectors[0], cluster_eigenvectors[i], match_method=match_method
        )
        cluster_orders.append(matching)
    cluster_orders = np.array(cluster_orders)
    
    correspondence_plot = get_correspondence_plot(
        images, cluster_eigenvectors, cluster_orders, discrete_colors, hw=16 * 2, n_cols=10
    )
    return correspondence_plot


def perform_two_image_interpolation(image1: Image.Image, 
                                  image2: Image.Image,
                                  model: CompressionModel, 
                                  interpolation_weights: List[float],
                                  n_clusters: int = 20, 
                                  match_method: str = 'hungarian',
                                  use_unit_norm: bool = False, 
                                  use_dino_matching: bool = True,
                                  use_multiscale_matching: bool = False,
                                  use_two_step_clustering: bool = False,
                                  n_superclusters: int = 3,
                                  n_subclusters_per_supercluster: int = 5,
                                  supercluster_match_method: str = 'hungarian',
                                  subcluster_match_method: str = 'hungarian',
                                  use_two_step_fgbg_clustering: bool = False,
                                  n_foreground_subclusters: int = 10,
                                  n_background_subclusters: int = 5,
                                  foreground_match_method: str = 'hungarian',
                                  background_match_method: str = 'hungarian',
                                  seed: Optional[int] = None,
                                  config_path: str = DEFAULT_CONFIG_PATH,
                                  return_matching: bool = False,
                                  predefined_matching = None) -> List[Image.Image]:
    """
    Interpolate between two images using the trained compression model.
    
    Args:
        image1, image2: Input PIL Images
        model: Trained compression model
        interpolation_weights: Weights for interpolation
        n_clusters: Number of clusters for correspondence matching (single-step only)
        match_method: Method for cluster matching (single-step only)
        use_unit_norm: Whether to normalize direction vectors
        use_dino_matching: Whether to use DINO-based matching or simple interpolation
        use_multiscale_matching: Whether to use multiscale matching (overrides 2-step if True)
        use_two_step_clustering: Whether to use 2-step hierarchical clustering
        n_superclusters: Number of coarse superclusters (2-step only)
        n_subclusters_per_supercluster: Number of subclusters per supercluster (2-step only)
        supercluster_match_method: Matching method for superclusters (2-step only)
        subcluster_match_method: Matching method for subclusters (2-step only)
        use_two_step_fgbg_clustering: Whether to use FG/BG 2-step clustering
        n_foreground_subclusters: Number of foreground subclusters (FG/BG only)
        n_background_subclusters: Number of background subclusters (FG/BG only)
        foreground_match_method: Matching method for foreground subclusters (FG/BG only)
        background_match_method: Matching method for background subclusters (FG/BG only)
        seed: Random seed for generation
        
    Returns:
        List[Image.Image]: Generated interpolated images
    """
    config = load_config(config_path)
    clear_gpu_memory()
    
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(img) for img in [image1, image2]])
    dino_image_embeds = extract_dino_features(images)
    compressed_image_embeds = model.encoder(dino_image_embeds)
    
    if use_multiscale_matching:
        direction_field = multiscale_directions(dino_image_embeds, compressed_image_embeds, n_cluster_list=[10, 30], n_repeats=1)
    elif use_two_step_fgbg_clustering and use_dino_matching:
        # Use FG/BG 2-step clustering
        supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster, fg_indices = kway_cluster_per_image_two_step_fgbg(
            dino_image_embeds,
            n_foreground_subclusters=n_foreground_subclusters,
            n_background_subclusters=n_background_subclusters
        )
        
        # Match using FG/BG 2-step approach
        subcluster_mapping = match_centers_two_step_fgbg(
            dino_image_embeds[0], dino_image_embeds[1],
            subcluster_eigvecs[0], subcluster_eigvecs[1],
            subcluster_to_supercluster[0], subcluster_to_supercluster[1],
            n_background_subclusters, n_foreground_subclusters,
            background_match_method=background_match_method,
            foreground_match_method=foreground_match_method
        )
        
        # Compute direction using subclusters
        direction_field = compute_direction_from_two_images_two_step(
            compressed_image_embeds, 
            [subcluster_eigvecs[0], subcluster_eigvecs[1]], 
            subcluster_mapping, 
            use_unit_norm
        )
    elif use_two_step_clustering and use_dino_matching:
        # Use 2-step hierarchical clustering
        supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster = kway_cluster_per_image_two_step(
            dino_image_embeds,
            n_superclusters=n_superclusters,
            n_subclusters_per_supercluster=n_subclusters_per_supercluster
        )
        
        # Match using 2-step approach
        subcluster_mapping = match_centers_two_step(
            dino_image_embeds[0], dino_image_embeds[1],
            supercluster_eigvecs[0], supercluster_eigvecs[1],
            subcluster_eigvecs[0], subcluster_eigvecs[1],
            subcluster_to_supercluster[0], subcluster_to_supercluster[1],
            supercluster_match_method=supercluster_match_method,
            subcluster_match_method=subcluster_match_method
        )
        
        # Compute direction using subclusters
        direction_field = compute_direction_from_two_images_two_step(
            compressed_image_embeds, 
            [subcluster_eigvecs[0], subcluster_eigvecs[1]], 
            subcluster_mapping, 
            use_unit_norm
        )
    elif use_dino_matching:
        # Use correspondence-based direction
        if predefined_matching is None:
            cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
            
            a_to_b_mapping = match_centers_two_images(
                dino_image_embeds[0], dino_image_embeds[1],
                cluster_eigenvectors[0], cluster_eigenvectors[1], 
                match_method=match_method
            )
            if return_matching:
                return cluster_eigenvectors, a_to_b_mapping
        
        if predefined_matching is not None:
            cluster_eigenvectors, a_to_b_mapping = predefined_matching
        
        direction_field = compute_direction_from_two_images(
            compressed_image_embeds, cluster_eigenvectors, a_to_b_mapping, use_unit_norm
        )
    else:
        # Simple linear interpolation in compressed space
        direction_field = compressed_image_embeds[1] - compressed_image_embeds[0]
    
    # Generate interpolated images
    ip_model = load_ipadapter(version=config.ipadapter_version)
    
    generated_images = []
    for weight in interpolation_weights:
        interpolated_embedding = compressed_image_embeds[0] + direction_field * weight
        decompressed_embedding = model.decoder(interpolated_embedding)
        
        batch_images = generate_images_from_clip_embeddings(
            ip_model, decompressed_embedding, num_samples=1, seed=seed
        )
        generated_images.extend(batch_images)
    
    # Clean up
    del ip_model
    clear_gpu_memory()
    
    return generated_images


def perform_n_image_interpolation(
    image_list: List[Image.Image],
    base_image_idx: int, 
    model: CompressionModel,
    interpolation_weights: List[Union[np.ndarray, List[float], torch.Tensor]],
    n_clusters: int = 20,
    match_method: str = 'hungarian',
    use_dino_matching: bool = True,
    use_two_step_clustering: bool = False,
    n_superclusters: int = 3,
    n_subclusters_per_supercluster: int = 5,
    supercluster_match_method: str = 'hungarian',
    subcluster_match_method: str = 'hungarian',
    use_two_step_fgbg_clustering: bool = False,
    n_foreground_subclusters: int = 10,
    n_background_subclusters: int = 5,
    foreground_match_method: str = 'hungarian',
    background_match_method: str = 'hungarian',
    seed: Optional[int] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> List[Image.Image]:
    """Interpolate across N images by cluster-aware linear combinations.

    Args:
        image_list: Input PIL images to blend.
        base_image_idx: index of image from image_list that is used as interpolation starting point
        model: Trained compression model used for encoding/decoding.
        interpolation_weights: Sequence of weight arrays, one per output image.
        n_clusters: Number of clusters for correspondence matching (single-step only).
        match_method: Cluster matching strategy (single-step only).
        use_dino_matching: Toggle for correspondence-guided interpolation.
        use_two_step_clustering: Whether to use 2-step hierarchical clustering.
        n_superclusters: Number of coarse superclusters (2-step only).
        n_subclusters_per_supercluster: Number of subclusters per supercluster (2-step only).
        supercluster_match_method: Matching method for superclusters (2-step only).
        subcluster_match_method: Matching method for subclusters (2-step only).
        use_two_step_fgbg_clustering: Whether to use FG/BG 2-step clustering.
        n_foreground_subclusters: Number of foreground subclusters (FG/BG only).
        n_background_subclusters: Number of background subclusters (FG/BG only).
        foreground_match_method: Matching method for foreground subclusters (FG/BG only).
        background_match_method: Matching method for background subclusters (FG/BG only).
        seed: Optional random seed forwarded to the generator.
        config_path: Path to model configuration.

    Returns:
        List of generated interpolated images.
    """

    if model is None or model == []:
        raise ValueError("A trained model is required for interpolation")
    if not image_list or len(image_list) < 2:
        raise ValueError("Provide at least two images for interpolation")
    if not interpolation_weights:
        return []

    n_images = len(image_list)
    if not 0 <= base_image_idx < n_images:
        raise ValueError("base_image_idx must reference an image in image_list")

    config = load_config(config_path)
    clear_gpu_memory()

    processed_images = torch.stack([dino_image_transform(img) for img in image_list])
    dino_image_embeds = extract_dino_features(processed_images)
    compressed_image_embeds = model.encoder(dino_image_embeds)

    device = compressed_image_embeds.device
    dtype = compressed_image_embeds.dtype

    def _weight_to_array(weight_obj: Union[np.ndarray, List[float], torch.Tensor]) -> np.ndarray:
        if isinstance(weight_obj, torch.Tensor):
            return weight_obj.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.asarray(weight_obj, dtype=np.float32).reshape(-1)

    if not use_dino_matching:
        # Simple weighted average in compressed space.
        ip_model = load_ipadapter(version=config.ipadapter_version)
        generated_images = []

        for weight in interpolation_weights:
            weight_array = _weight_to_array(weight)
            if weight_array.size != n_images:
                raise ValueError("Each weight array must have length equal to the number of input images")
            weight_tensor = torch.as_tensor(weight_array, device=device, dtype=dtype)
            combined_embedding = torch.sum(
                compressed_image_embeds * weight_tensor.view(-1, 1, 1), dim=0
            )
            decoded_embedding = model.decoder(combined_embedding)
            batch_images = generate_images_from_clip_embeddings(
                ip_model, decoded_embedding, num_samples=1, seed=seed
            )
            generated_images.extend(batch_images)

        del ip_model
        clear_gpu_memory()
        return generated_images

    # Use DINO matching
    if use_two_step_fgbg_clustering:
        # FG/BG 2-step clustering
        supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster, fg_indices = kway_cluster_per_image_two_step_fgbg(
            dino_image_embeds,
            n_foreground_subclusters=n_foreground_subclusters,
            n_background_subclusters=n_background_subclusters
        )
        
        cluster_eigenvectors = subcluster_eigvecs
        effective_n_clusters = n_foreground_subclusters + n_background_subclusters
        
        # Match every image's subclusters back to the base image
        cluster_mappings: List[np.ndarray] = [np.zeros(effective_n_clusters, dtype=np.int64) for _ in range(n_images)]
        for image_idx in range(n_images):
            if image_idx == base_image_idx:
                cluster_mappings[image_idx] = np.arange(effective_n_clusters)
            else:
                cluster_mappings[image_idx] = match_centers_two_step_fgbg(
                    dino_image_embeds[base_image_idx],
                    dino_image_embeds[image_idx],
                    subcluster_eigvecs[base_image_idx],
                    subcluster_eigvecs[image_idx],
                    subcluster_to_supercluster[base_image_idx],
                    subcluster_to_supercluster[image_idx],
                    n_background_subclusters,
                    n_foreground_subclusters,
                    background_match_method=background_match_method,
                    foreground_match_method=foreground_match_method,
                )
    elif use_two_step_clustering:
        # 2-step hierarchical clustering
        supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster = kway_cluster_per_image_two_step(
            dino_image_embeds,
            n_superclusters=n_superclusters,
            n_subclusters_per_supercluster=n_subclusters_per_supercluster
        )
        
        cluster_eigenvectors = subcluster_eigvecs
        effective_n_clusters = n_superclusters * n_subclusters_per_supercluster
        
        # Match every image's subclusters back to the base image
        cluster_mappings: List[np.ndarray] = [np.zeros(effective_n_clusters, dtype=np.int64) for _ in range(n_images)]
        for image_idx in range(n_images):
            if image_idx == base_image_idx:
                cluster_mappings[image_idx] = np.arange(effective_n_clusters)
            else:
                cluster_mappings[image_idx] = match_centers_two_step(
                    dino_image_embeds[base_image_idx],
                    dino_image_embeds[image_idx],
                    supercluster_eigvecs[base_image_idx],
                    supercluster_eigvecs[image_idx],
                    subcluster_eigvecs[base_image_idx],
                    subcluster_eigvecs[image_idx],
                    subcluster_to_supercluster[base_image_idx],
                    subcluster_to_supercluster[image_idx],
                    supercluster_match_method=supercluster_match_method,
                    subcluster_match_method=subcluster_match_method,
                )
    else:
        # Single-step clustering
        cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
        effective_n_clusters = n_clusters
        
        # Match every image's clusters back to the chosen base image.
        cluster_mappings: List[np.ndarray] = [np.zeros(effective_n_clusters, dtype=np.int64) for _ in range(n_images)]
        for image_idx in range(n_images):
            if image_idx == base_image_idx:
                cluster_mappings[image_idx] = np.arange(effective_n_clusters)
            else:
                cluster_mappings[image_idx] = match_centers_two_images(
                    dino_image_embeds[base_image_idx],
                    dino_image_embeds[image_idx],
                    cluster_eigenvectors[base_image_idx],
                    cluster_eigenvectors[image_idx],
                    match_method=match_method,
                )

    cluster_labels = [eig.argmax(-1).cpu() for eig in cluster_eigenvectors]
    cluster_centers = [
        get_cluster_center_features(compressed_image_embeds[i], cluster_labels[i], effective_n_clusters)
        for i in range(n_images)
    ]

    base_embedding = compressed_image_embeds[base_image_idx]
    base_cluster_labels = cluster_labels[base_image_idx]
    embed_dim = compressed_image_embeds.shape[-1]

    ip_model = load_ipadapter(version=config.ipadapter_version)
    generated_images: List[Image.Image] = []

    for weight in interpolation_weights:
        weight_array = _weight_to_array(weight)
        if weight_array.size != n_images:
            raise ValueError("Each weight array must have length equal to the number of input images")

        weight_tensor = torch.as_tensor(weight_array, device=device, dtype=dtype)
        blended_embedding = base_embedding.clone()

        for cluster_idx in range(effective_n_clusters):
            base_mask = base_cluster_labels == cluster_idx
            if base_mask.sum() == 0:
                continue

            base_mask_device = base_mask.to(device=device)
            base_center = cluster_centers[base_image_idx][cluster_idx].to(device=device, dtype=dtype)
            direction_vector = torch.zeros((embed_dim,), device=device, dtype=dtype)

            for image_idx in range(n_images):
                mapped_cluster = int(cluster_mappings[image_idx][cluster_idx])
                image_mask = cluster_labels[image_idx] == mapped_cluster
                if image_mask.sum() == 0:
                    continue

                center_feat = cluster_centers[image_idx][mapped_cluster].to(device=device, dtype=dtype)
                direction_vector = direction_vector + (center_feat - base_center) * weight_tensor[image_idx]

            blended_embedding[base_mask_device] = base_embedding[base_mask_device] + direction_vector

        decoded_embedding = model.decoder(blended_embedding)
        batch_images = generate_images_from_clip_embeddings(
            ip_model, decoded_embedding, num_samples=1, seed=seed
        )
        generated_images.extend(batch_images)

    del ip_model
    clear_gpu_memory()

    return generated_images


def perform_n_image_interpolation_per_cluster(
    image_list: List[Image.Image],
    base_image_idx: int,
    model: CompressionModel,
    interpolation_weights: List[Union[np.ndarray, List[float], torch.Tensor]],
    n_clusters: int = 20,
    match_method: str = 'hungarian',
    use_dino_matching: bool = True,
    use_two_step_clustering: bool = False,
    n_superclusters: int = 3,
    n_subclusters_per_supercluster: int = 5,
    supercluster_match_method: str = 'hungarian',
    subcluster_match_method: str = 'hungarian',
    use_two_step_fgbg_clustering: bool = False,
    n_foreground_subclusters: int = 10,
    n_background_subclusters: int = 5,
    foreground_match_method: str = 'hungarian',
    background_match_method: str = 'hungarian',
    seed: Optional[int] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    precomputed_dino_embeds: Optional[torch.Tensor] = None,
    precomputed_cluster_eigenvectors: Optional[torch.Tensor] = None,
    precomputed_cluster_mappings: Optional[List[np.ndarray]] = None,
) -> List[Image.Image]:
    """Blend directions per cluster with cluster-specific weights.

    Args mirror ``perform_n_image_interpolation`` but ``interpolation_weights`` must
    contain exactly ``n_clusters`` entries (or n_superclusters * n_subclusters_per_supercluster
    if using 2-step clustering, or n_foreground_subclusters + n_background_subclusters
    if using FG/BG clustering). Each entry is a weight array that gauges how strongly 
    to follow the base-to-image direction for the corresponding cluster. Only one output 
    image is generated.
    
    If precomputed_dino_embeds, precomputed_cluster_eigenvectors, and 
    precomputed_cluster_mappings are provided, they will be used instead of 
    recomputing clusters, ensuring consistency with UI cluster visualization.
    
    Additional args:
        use_two_step_clustering: Whether to use 2-step hierarchical clustering.
        n_superclusters: Number of coarse superclusters (2-step only).
        n_subclusters_per_supercluster: Number of subclusters per supercluster (2-step only).
        supercluster_match_method: Matching method for superclusters (2-step only).
        subcluster_match_method: Matching method for subclusters (2-step only).
        use_two_step_fgbg_clustering: Whether to use FG/BG 2-step clustering.
        n_foreground_subclusters: Number of foreground subclusters (FG/BG only).
        n_background_subclusters: Number of background subclusters (FG/BG only).
        foreground_match_method: Matching method for foreground subclusters (FG/BG only).
        background_match_method: Matching method for background subclusters (FG/BG only).
    """

    if model is None or model == []:
        raise ValueError("A trained model is required for interpolation")
    if not image_list or len(image_list) < 2:
        raise ValueError("Provide at least two images for interpolation")
    if not 0 <= base_image_idx < len(image_list):
        raise ValueError("base_image_idx must reference an image in image_list")
    
    # Determine effective number of clusters
    if use_two_step_fgbg_clustering:
        effective_n_clusters = n_foreground_subclusters + n_background_subclusters
    elif use_two_step_clustering:
        effective_n_clusters = n_superclusters * n_subclusters_per_supercluster
    else:
        effective_n_clusters = n_clusters
    
    if len(interpolation_weights) != effective_n_clusters:
        raise ValueError(f"interpolation_weights length must match effective number of clusters ({effective_n_clusters})")

    n_images = len(image_list)

    if not use_dino_matching:
        raise ValueError("Cluster-wise interpolation requires DINO matching")

    config = load_config(config_path)
    clear_gpu_memory()

    processed_images = torch.stack([dino_image_transform(img) for img in image_list])
    
    # Use precomputed values if available, otherwise compute from scratch
    if precomputed_dino_embeds is not None:
        dino_image_embeds = precomputed_dino_embeds
    else:
        dino_image_embeds = extract_dino_features(processed_images)
    
    compressed_image_embeds = model.encoder(dino_image_embeds)

    device = compressed_image_embeds.device
    dtype = compressed_image_embeds.dtype

    def _weight_to_array(weight_obj: Union[np.ndarray, List[float], torch.Tensor]) -> np.ndarray:
        if isinstance(weight_obj, torch.Tensor):
            return weight_obj.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.asarray(weight_obj, dtype=np.float32).reshape(-1)

    # Compute or use precomputed clusters
    if precomputed_cluster_eigenvectors is not None:
        cluster_eigenvectors = precomputed_cluster_eigenvectors
    else:
        if use_two_step_fgbg_clustering:
            # FG/BG 2-step clustering
            supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster, fg_indices = kway_cluster_per_image_two_step_fgbg(
                dino_image_embeds,
                n_foreground_subclusters=n_foreground_subclusters,
                n_background_subclusters=n_background_subclusters
            )
            cluster_eigenvectors = subcluster_eigvecs
        elif use_two_step_clustering:
            # 2-step hierarchical clustering
            supercluster_eigvecs, subcluster_eigvecs, subcluster_to_supercluster = kway_cluster_per_image_two_step(
                dino_image_embeds,
                n_superclusters=n_superclusters,
                n_subclusters_per_supercluster=n_subclusters_per_supercluster
            )
            cluster_eigenvectors = subcluster_eigvecs
        else:
            # Single-step clustering
            cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
    
    cluster_labels = [eig.argmax(-1).cpu() for eig in cluster_eigenvectors]

    # Compute or use precomputed mappings
    if precomputed_cluster_mappings is not None:
        cluster_mappings = precomputed_cluster_mappings
    else:
        cluster_mappings: List[np.ndarray] = [np.zeros(effective_n_clusters, dtype=np.int64) for _ in range(n_images)]
        for image_idx in range(n_images):
            if image_idx == base_image_idx:
                cluster_mappings[image_idx] = np.arange(effective_n_clusters)
            else:
                if use_two_step_fgbg_clustering:
                    cluster_mappings[image_idx] = match_centers_two_step_fgbg(
                        dino_image_embeds[base_image_idx],
                        dino_image_embeds[image_idx],
                        subcluster_eigvecs[base_image_idx],
                        subcluster_eigvecs[image_idx],
                        subcluster_to_supercluster[base_image_idx],
                        subcluster_to_supercluster[image_idx],
                        n_background_subclusters,
                        n_foreground_subclusters,
                        background_match_method=background_match_method,
                        foreground_match_method=foreground_match_method,
                    )
                elif use_two_step_clustering:
                    cluster_mappings[image_idx] = match_centers_two_step(
                        dino_image_embeds[base_image_idx],
                        dino_image_embeds[image_idx],
                        supercluster_eigvecs[base_image_idx],
                        supercluster_eigvecs[image_idx],
                        subcluster_eigvecs[base_image_idx],
                        subcluster_eigvecs[image_idx],
                        subcluster_to_supercluster[base_image_idx],
                        subcluster_to_supercluster[image_idx],
                        supercluster_match_method=supercluster_match_method,
                        subcluster_match_method=subcluster_match_method,
                    )
                else:
                    cluster_mappings[image_idx] = match_centers_two_images(
                        dino_image_embeds[base_image_idx],
                        dino_image_embeds[image_idx],
                        cluster_eigenvectors[base_image_idx],
                        cluster_eigenvectors[image_idx],
                        match_method=match_method,
                    )

    cluster_centers = [
        get_cluster_center_features(compressed_image_embeds[i], cluster_labels[i], effective_n_clusters)
        for i in range(n_images)
    ]

    base_embedding = compressed_image_embeds[base_image_idx]
    base_cluster_labels = cluster_labels[base_image_idx]
    embed_dim = compressed_image_embeds.shape[-1]

    blended_embedding = base_embedding.clone()

    for cluster_idx in range(effective_n_clusters):
        base_mask = base_cluster_labels == cluster_idx
        if base_mask.sum() == 0:
            continue

        weight_array = _weight_to_array(interpolation_weights[cluster_idx])
        if weight_array.size != n_images:
            raise ValueError("Each per-cluster weight array must match number of images")

        weight_tensor = torch.as_tensor(weight_array, device=device, dtype=dtype)

        base_center = cluster_centers[base_image_idx][cluster_idx].to(device=device, dtype=dtype)
        direction_vector = torch.zeros((embed_dim,), device=device, dtype=dtype)

        for image_idx in range(n_images):
            mapped_cluster = int(cluster_mappings[image_idx][cluster_idx])
            cluster_mask = cluster_labels[image_idx] == mapped_cluster
            if cluster_mask.sum() == 0:
                continue

            center_feat = cluster_centers[image_idx][mapped_cluster].to(device=device, dtype=dtype)
            direction_vector = direction_vector + (center_feat - base_center) * weight_tensor[image_idx]

        base_mask_device = base_mask.to(device=device)
        blended_embedding[base_mask_device] = base_embedding[base_mask_device] + direction_vector

    decoded_embedding = model.decoder(blended_embedding)

    ip_model = load_ipadapter(version=config.ipadapter_version)
    try:
        generated_images = generate_images_from_clip_embeddings(
            ip_model, decoded_embedding, num_samples=1, seed=seed
        )
    finally:
        del ip_model
        clear_gpu_memory()

    return generated_images

def interpolate_two_images_no_compression(image1: Image.Image, image2: Image.Image, interpolation_weights: List[float], n_clusters: int = 20, match_method: str = 'hungarian', 
                                          use_unit_norm: bool = False, use_multiscale_matching: bool = False, dino_matching: bool = True, seed: Optional[int] = None, config_path: str = DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    clip_images = torch.stack([clip_image_transform(image) for image in [image1, image2]])
    dino_images = torch.stack([dino_image_transform(image) for image in [image1, image2]])
    # downsample to 256x256
    dino_images = torch.nn.functional.interpolate(dino_images, size=(256, 256), mode='bilinear', align_corners=False)
    dino_image_embeds = extract_dino_features(dino_images)
    clip_image_embeds = extract_clip_features(clip_images, ipadapter_version=config.ipadapter_version)
    input_embeds = dino_image_embeds

    b, l, c = input_embeds.shape
    joint_eigvecs, joint_rgbs = ncut_tsne_multiple_images(input_embeds, n_eig=30, gamma=None)
    single_eigvecs = kway_cluster_per_image(input_embeds, n_clusters=n_clusters, gamma=None)

    A_to_B = match_centers_two_images(dino_image_embeds[0], dino_image_embeds[1], single_eigvecs[0], single_eigvecs[1], match_method=match_method)

    if use_multiscale_matching:
        direction = multiscale_directions(dino_image_embeds, clip_image_embeds, n_cluster_list=[10, 30], n_repeats=1)
    elif dino_matching:
        direction = compute_direction_from_two_images(clip_image_embeds, single_eigvecs, A_to_B, use_unit_norm=use_unit_norm)
    else:
        direction = clip_image_embeds[1] - clip_image_embeds[0]

    ip_model = load_ipadapter(version=config.ipadapter_version)
    
    interpolated_images = []
    for w in interpolation_weights:
        A_interpolated = clip_image_embeds[0] + direction * w
        gen_images = generate_images_from_clip_embeddings(ip_model, A_interpolated, num_samples=1, seed=seed)
        interpolated_images.extend(gen_images)
    
    return interpolated_images


# Apply HuggingFace Spaces GPU decorator if available
if USE_HUGGINGFACE_ZEROGPU:
    perform_three_image_analogy = spaces.GPU(duration=60)(perform_three_image_analogy)
    perform_two_image_interpolation = spaces.GPU(duration=60)(perform_two_image_interpolation)
    interpolate_two_images_no_compression = spaces.GPU(duration=60)(interpolate_two_images_no_compression)


# ===== Visualization Functions =====

def plot_training_loss(model: CompressionModel) -> plt.Figure:
    """
    Create a plot showing training loss curves.
    
    Args:
        model: Trained compression model with loss history
        
    Returns:
        matplotlib.pyplot.Figure: Loss plot figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Reconstruction loss
    if 'recon' in model.loss_history and model.loss_history['recon']:
        ax1.plot(model.loss_history['recon'], 'b-', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Reconstruction Loss')
        ax1.set_title('Reconstruction Loss')
        ax1.grid(True, alpha=0.3)

    # Encoder flag loss
    if 'flag_encoder' in model.loss_history and model.loss_history['flag_encoder']:
        flag_loss = np.array(model.loss_history['flag_encoder'])
        ax2.plot(flag_loss, 'r-', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Encoder Flag Loss')
        ax2.set_title('Encoder Flag Loss')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ===== Gradio Interface Setup =====

def create_gradio_interface():
    """Create and configure the main Gradio interface."""
    MAX_INTERP_IMAGES = 10
    MAX_INTERP_CLUSTERS = 20

    # Load default images
    DEFAULT_IMAGES = load_default_images()
    
    # Download models if using HuggingFace Spaces
    if USE_HUGGINGFACE_ZEROGPU:
        try:
            from download_models import download_ipadapter
            download_ipadapter()
        except ImportError:
            logging.warning("Could not import download_models")

    # Create Gradio interface
    theme = gr.themes.Base(
        spacing_size='md', 
        text_size='lg', 
        primary_hue='blue', 
        neutral_hue='slate', 
        secondary_hue='pink'
    )
    
    demo = gr.Blocks(theme=theme)
    
    with demo:
        # Shared state for trained model
        model_state = gr.State([])

        # ===== Tab 1: Model Training =====
        with gr.Tab("1. Mood Space Training"):
            gr.Markdown("""
            ## Mood Space Training
            
            Upload images to train a neural compression model that learns a meaningful 
            "mood space" representation. The model will learn to compress high-dimensional 
            visual features while preserving semantic and geometric relationships.
            
            **Instructions:**
            1. Upload 2-10 images that share some thematic relationship
            2. Adjust training parameters if needed
            3. Click "Train" to start the training process
            4. Use the trained model in the other tabs for interpolation and analogies
            """)

            with gr.Row():
                with gr.Column():
                    # Image upload interface
                    input_images = gr.Gallery(
                        label="Training Images", 
                        show_label=True,
                        columns=3,
                        rows=2,
                        height=400
                    )
                    
                    upload_button = gr.UploadButton(
                        label="Upload Images", 
                        variant='secondary', 
                        file_types=["image"], 
                        file_count="multiple"
                    )
                    
                    def add_uploaded_images(existing_images, new_images):
                        """Add newly uploaded images to the gallery."""
                        if existing_images is None:
                            existing_images = []
                        if new_images is None:
                            return existing_images
                        
                        # Process new images
                        if isinstance(new_images, list):
                            new_pil_images = [Image.open(img) for img in new_images]
                        else:
                            new_pil_images = [Image.open(new_images)]
                        
                        existing_images.extend(new_pil_images)
                        gr.Info(f"Added {len(new_pil_images)} images. Total: {len(existing_images)}")
                        return existing_images
                    
                    upload_button.upload(
                        add_uploaded_images, 
                        inputs=[input_images, upload_button], 
                        outputs=[input_images]
                    )

                with gr.Column():
                    # Training parameters
                    with gr.Accordion("Training Parameters", open=False):
                        lr_slider = gr.Slider(
                            minimum=0.0001, maximum=0.01, step=0.0001, value=0.001, visible=False,
                            label="Learning Rate", info="Higher values train faster but may be unstable"
                        )
                        steps_slider = gr.Slider(
                            minimum=1000, maximum=100000, step=100, value=1000, visible=True,
                            label="Training Steps", info="More steps = better quality but slower"
                        )
                        width_slider = gr.Slider(
                            minimum=16, maximum=4096, step=16, value=512, visible=False,
                            label="MLP Width", info="Network capacity"
                        )
                        layers_slider = gr.Slider(
                            minimum=1, maximum=8, step=1, value=4, visible=False,
                            label="MLP Layers", info="Network depth"
                        )
                        n_eig_slider = gr.Slider(
                            minimum=8, maximum=64, step=8, value=64, visible=True,
                            label="Number of Eigenvectors", info="Number of eigenvectors to used for flag loss, 8 for coarse blend, 64 for fine blend"
                        )
                    
                    train_button = gr.Button("Train Model", variant="primary", size="lg")
                    
                    # Training wrapper function
                    def training_wrapper(images, lr, steps, width, layers, n_eig):
                        """Wrapper for training that handles UI feedback."""
                        if not images or len(images) < 2:
                            gr.Error("Please upload at least 2 images for training")
                            return None, None
                        
                        try:
                            model, trainer = train_mood_space(images, lr, steps, width, layers, n_eig)
                            loss_plot = plot_training_loss(model)
                            gr.Info("Training completed successfully!")
                            return model, loss_plot
                        except Exception as e:
                            gr.Error(f"Training failed: {str(e)}")
                            return None, None

                    loss_plot = gr.Plot(label="Training Progress")
                    train_button.click(
                        training_wrapper,
                        inputs=[input_images, lr_slider, steps_slider, width_slider, layers_slider, n_eig_slider],
                        outputs=[model_state, loss_plot]
                    )

            # Example image sets
            gr.Markdown("## Example Image Sets")
            example_sets = {
                "Bears (Rotation)": ["./images/black_bear1.jpg", "./images/black_bear2.jpg"],
                "Bear Analogy": ["./images/black_bear1.jpg", "./images/black_bear2.jpg", "./images/pink_bear1.jpg"],
                "Dog → Fish": ["./images/dog1.jpg", "./images/fish.jpg"],
                "Duck → Paper": ["./images/duck1.jpg", "./images/toilet_paper.jpg"],
                "Portrait → Action": ["./images/jimi_portrait.jpg", "./images/jimi_action.jpg", "./images/bach_portrait.jpg"],
            }
            
            for set_name, image_paths in example_sets.items():
                with gr.Row():
                    with gr.Column(scale=2):
                        add_btn = gr.Button(f"Load {set_name}", size="sm")
                    with gr.Column(scale=8):
                        example_gallery = gr.Gallery(
                            value=image_paths,
                            columns=len(image_paths),
                            rows=1,
                            height=200,
                            show_label=False
                        )
                    
                    def load_example_set(gallery_images):
                        return [img[0] if isinstance(img, tuple) else img for img in gallery_images]
                    
                    add_btn.click(
                        load_example_set,
                        inputs=[example_gallery],
                        outputs=[input_images]
                    )

        # ===== Tab 2: Two-Image Interpolation =====
        with gr.Tab("2. Image Interpolation"):
            gr.Markdown("""
            ## Two-Image Interpolation
            
            Smoothly interpolate between two images using the trained Mood Space model.
            The model finds semantic correspondences between image regions and creates
            meaningful transitions.
            """)

            with gr.Row():
                image_a = gr.Image(label="Image A", type="pil")
                image_b = gr.Image(label="Image B", type="pil")
                
                with gr.Column():
                    reload_btn = gr.Button("Load from Training Set")
                    
                    with gr.Accordion("Interpolation Settings", open=False):
                        w_start = gr.Slider(minimum=-2, maximum=2, step=0.1, value=0, label="Start Weight")
                        w_end = gr.Slider(minimum=-2, maximum=2, step=0.1, value=1, label="End Weight")
                        n_steps = gr.Slider(minimum=3, maximum=20, step=1, value=10, label="Number of Steps")
                        
                        # Clustering method selection
                        clustering_method = gr.Radio(
                            choices=["Single-step", "Hierarchical 2-step", "FG/BG 2-step"],
                            value="Single-step",
                            label="Clustering Method",
                            info="Choose between single-step, hierarchical, or foreground/background clustering"
                        )
                        
                        # Single-step clustering controls
                        with gr.Group(visible=True) as single_step_controls:
                            n_clusters = gr.Slider(minimum=5, maximum=50, step=1, value=10, label="Number of Clusters")
                            match_method = gr.Radio(
                                ["hungarian", "argmin"],
                                value="hungarian",
                                label="Matching Method"
                            )
                        
                        # Hierarchical two-step clustering controls
                        with gr.Group(visible=False) as hierarchical_two_step_controls:
                            n_superclusters = gr.Slider(
                                minimum=2, maximum=10, step=1, value=3,
                                label="Number of Superclusters",
                                info="Coarse clusters found first"
                            )
                            n_subclusters_per_supercluster = gr.Slider(
                                minimum=2, maximum=10, step=1, value=5,
                                label="Subclusters per Supercluster",
                                info="Fine clusters within each supercluster"
                            )
                            supercluster_match_method = gr.Radio(
                                ["hungarian", "argmin"],
                                value="hungarian",
                                label="Supercluster Matching Method"
                            )
                            subcluster_match_method = gr.Radio(
                                ["hungarian", "argmin"],
                                value="hungarian",
                                label="Subcluster Matching Method"
                            )
                        
                        # FG/BG two-step clustering controls
                        with gr.Group(visible=False) as fgbg_two_step_controls:
                            n_foreground_subclusters = gr.Slider(
                                minimum=2, maximum=20, step=1, value=10,
                                label="Foreground Subclusters",
                                info="Number of clusters for foreground region"
                            )
                            n_background_subclusters = gr.Slider(
                                minimum=2, maximum=20, step=1, value=5,
                                label="Background Subclusters",
                                info="Number of clusters for background region"
                            )
                            foreground_match_method = gr.Radio(
                                ["hungarian", "argmin"],
                                value="hungarian",
                                label="Foreground Matching Method"
                            )
                            background_match_method = gr.Radio(
                                ["hungarian", "argmin"],
                                value="hungarian",
                                label="Background Matching Method"
                            )
                        
                        # Toggle visibility based on clustering method
                        def toggle_clustering_controls(method):
                            return (
                                gr.update(visible=(method == "Single-step")),
                                gr.update(visible=(method == "Hierarchical 2-step")),
                                gr.update(visible=(method == "FG/BG 2-step"))
                            )
                        
                        clustering_method.change(
                            toggle_clustering_controls,
                            inputs=[clustering_method],
                            outputs=[single_step_controls, hierarchical_two_step_controls, fgbg_two_step_controls]
                        )
                    
                    interpolate_btn = gr.Button("Interpolate", variant="primary")

            interpolation_result = gr.Gallery(
                label="Interpolation Results", 
                columns=5, 
                rows=2
            )
            
            # Interpolation function
            def run_interpolation(
                img_a, img_b, model, w_start, w_end, n_steps, 
                clustering_method_choice,
                n_clusters, match_method,
                n_superclusters, n_subclusters, 
                supercluster_match, subcluster_match,
                n_fg_subclusters, n_bg_subclusters,
                fg_match, bg_match
            ):
                if model is None or model == []:
                    gr.Error("Please train a model first")
                    return None
                
                if img_a is None or img_b is None:
                    gr.Error("Please provide both input images")
                    return None
                
                weights = torch.linspace(w_start, w_end, n_steps).tolist()
                
                # Determine which clustering method to use
                use_two_step = (clustering_method_choice == "Hierarchical 2-step")
                use_fgbg = (clustering_method_choice == "FG/BG 2-step")
                
                result_images = perform_two_image_interpolation(
                    img_a, img_b, model, weights, 
                    n_clusters=n_clusters, 
                    match_method=match_method,
                    use_two_step_clustering=use_two_step,
                    n_superclusters=n_superclusters,
                    n_subclusters_per_supercluster=n_subclusters,
                    supercluster_match_method=supercluster_match,
                    subcluster_match_method=subcluster_match,
                    use_two_step_fgbg_clustering=use_fgbg,
                    n_foreground_subclusters=n_fg_subclusters,
                    n_background_subclusters=n_bg_subclusters,
                    foreground_match_method=fg_match,
                    background_match_method=bg_match
                )
                
                # Resize for display
                display_images = [
                    img.resize((256, 256), Image.Resampling.LANCZOS) 
                    for img in result_images
                ]
                
                return display_images
            
            interpolate_btn.click(
                run_interpolation,
                inputs=[
                    image_a, image_b, model_state, w_start, w_end, n_steps, 
                    clustering_method,
                    n_clusters, match_method,
                    n_superclusters, n_subclusters_per_supercluster,
                    supercluster_match_method, subcluster_match_method,
                    n_foreground_subclusters, n_background_subclusters,
                    foreground_match_method, background_match_method
                ],
                outputs=[interpolation_result]
            )
            
            # Auto-load from training images
            def load_first_two_images(training_images):
                if training_images and len(training_images) >= 2:
                    processed_images = load_gradio_images_helper(training_images)
                    if len(processed_images) >= 2:
                        return processed_images[0], processed_images[1]
                return None, None
            
            reload_btn.click(
                load_first_two_images,
                inputs=[input_images],
                outputs=[image_a, image_b]
            )

        # ===== Tab 3: Three-Image Analogy =====
        with gr.Tab("3. Visual Analogy"):
            gr.Markdown("""
            ## Visual Analogy (Path Lifting)
            
            Perform visual analogies: Given A1 → B1, predict A2 → B2.
            This demonstrates the model's ability to understand and transfer 
            visual transformations between different objects.
            """)

            with gr.Row():
                img_a1 = gr.Image(label="A1 (Source)", type="pil")
                img_b1 = gr.Image(label="B1 (Target)", type="pil") 
                img_a2 = gr.Image(label="A2 (Query)", type="pil")
                predicted_b2 = gr.Image(label="B2 (Predicted)", type="pil", interactive=False)
                
                with gr.Column():
                    reload_btn_3 = gr.Button("Load from Training Set")
                    
                    with gr.Accordion("Analogy Settings", open=False):
                        analogy_w_start = gr.Slider(minimum=-2, maximum=2, step=0.1, value=0, label="Start Weight")
                        analogy_w_end = gr.Slider(minimum=-2, maximum=2, step=0.1, value=1, label="End Weight") 
                        analogy_n_steps = gr.Slider(minimum=3, maximum=20, step=1, value=10, label="Number of Steps")
                        analogy_n_clusters = gr.Slider(minimum=5, maximum=50, step=1, value=10, label="Clusters for Matching")
                        analogy_match_method = gr.Radio(
                            ["hungarian", "argmin"],
                            value="hungarian",
                            label="Matching Method"
                        )
                    
                    analogy_btn = gr.Button("Run Analogy", variant="primary")

            # Analogy Results
            analogy_plot = gr.Plot(label="Analogy Results")
            
            # Correspondence Visualization
            correspondence_plot = gr.Image(label="Correspondence Visualization")
            
            analogy_results = gr.Gallery(
                label="Generated Sequence",
                columns=6,
                rows=2,
                visible=False
            )
            
            # Analogy function  
            def run_analogy(a1, b1, a2, model, w_start, w_end, n_steps, n_clusters, match_method):
                if model is None or model == []:
                    gr.Error("Please train a model first")
                    return None, None, None
                
                if a1 is None or b1 is None or a2 is None:
                    gr.Error("Please provide all three input images")
                    return None, None, None
                
                weights = torch.linspace(w_start, w_end, n_steps).tolist()
                correspondence_img, result_plot, result_images = perform_three_image_analogy(
                    [a2, a1, b1], model, weights, n_clusters, 1, match_method
                )
                
                return result_plot, correspondence_img, result_images
            
            analogy_btn.click(
                run_analogy,
                inputs=[img_a1, img_b1, img_a2, model_state, analogy_w_start, analogy_w_end, 
                       analogy_n_steps, analogy_n_clusters, analogy_match_method],
                outputs=[analogy_plot, correspondence_plot, analogy_results]
            )
            
            # Auto-load from training images
            def load_three_images(training_images):
                if training_images and len(training_images) >= 3:
                    processed_images = load_gradio_images_helper(training_images)
                    if len(processed_images) >= 3:
                        return processed_images[0], processed_images[1], processed_images[2]
                    elif len(processed_images) >= 2:
                        return processed_images[0], processed_images[1], processed_images[0]
                elif training_images and len(training_images) >= 2:
                    processed_images = load_gradio_images_helper(training_images)
                    if len(processed_images) >= 2:
                        return processed_images[0], processed_images[1], processed_images[0]
                return None, None, None
            
            reload_btn_3.click(
                load_three_images,
                inputs=[input_images],
                outputs=[img_a1, img_b1, img_a2]
            )

        # ===== Tab 4: Clustered N-Image Interpolation =====
        with gr.Tab("4. N-Image Interpolation"):
            gr.Markdown("""
            ## Cluster-Aware N-Image Interpolation

            Combine multiple training images by assigning per-cluster weights to each
            image. This workflow lets you choose a base image, inspect the discovered
            correspondences, and fine-tune how much influence every image has on each
            cluster before generating a single blended result.
            """)

            cluster_config_state = gr.State({})
            cluster_weight_state = gr.State([])

            with gr.Row():
                base_selector = gr.Dropdown(
                    label="Base Image",
                    choices=[],
                    value=None,
                    interactive=True
                )
                
            with gr.Row():
                # Clustering method selection
                clustering_method_nimage = gr.Radio(
                    choices=["Single-step", "Hierarchical 2-step", "FG/BG 2-step"],
                    value="Single-step",
                    label="Clustering Method",
                    info="Choose between single-step, hierarchical, or foreground/background clustering"
                )
                
            # Single-step clustering controls
            with gr.Group(visible=True) as single_step_controls_nimage:
                with gr.Row():
                    cluster_count_slider = gr.Slider(
                        minimum=2,
                        maximum=MAX_INTERP_CLUSTERS,
                        step=1,
                        value=6,
                        label="Number of Clusters"
                    )
                    match_method_dropdown = gr.Radio(
                        ["hungarian", "argmin"],
                        value="hungarian",
                        label="Matching Method"
                    )
            
            # Hierarchical two-step clustering controls
            with gr.Group(visible=False) as hierarchical_two_step_controls_nimage:
                with gr.Row():
                    n_superclusters_nimage = gr.Slider(
                        minimum=2, maximum=10, step=1, value=3,
                        label="Number of Superclusters",
                        info="Coarse clusters found first"
                    )
                    n_subclusters_per_supercluster_nimage = gr.Slider(
                        minimum=2, maximum=10, step=1, value=5,
                        label="Subclusters per Supercluster",
                        info="Fine clusters within each supercluster"
                    )
                with gr.Row():
                    supercluster_match_method_nimage = gr.Radio(
                        ["hungarian", "argmin"],
                        value="hungarian",
                        label="Supercluster Matching Method"
                    )
                    subcluster_match_method_nimage = gr.Radio(
                        ["hungarian", "argmin"],
                        value="hungarian",
                        label="Subcluster Matching Method"
                    )
            
            # FG/BG two-step clustering controls
            with gr.Group(visible=False) as fgbg_two_step_controls_nimage:
                with gr.Row():
                    n_foreground_subclusters_nimage = gr.Slider(
                        minimum=2, maximum=20, step=1, value=10,
                        label="Foreground Subclusters",
                        info="Number of clusters for foreground region"
                    )
                    n_background_subclusters_nimage = gr.Slider(
                        minimum=2, maximum=20, step=1, value=5,
                        label="Background Subclusters",
                        info="Number of clusters for background region"
                    )
                with gr.Row():
                    foreground_match_method_nimage = gr.Radio(
                        ["hungarian", "argmin"],
                        value="hungarian",
                        label="Foreground Matching Method"
                    )
                    background_match_method_nimage = gr.Radio(
                        ["hungarian", "argmin"],
                        value="hungarian",
                        label="Background Matching Method"
                    )
            
            # Toggle visibility based on clustering method
            def toggle_clustering_controls_nimage(method):
                return (
                    gr.update(visible=(method == "Single-step")),
                    gr.update(visible=(method == "Hierarchical 2-step")),
                    gr.update(visible=(method == "FG/BG 2-step"))
                )
            
            clustering_method_nimage.change(
                toggle_clustering_controls_nimage,
                inputs=[clustering_method_nimage],
                outputs=[single_step_controls_nimage, hierarchical_two_step_controls_nimage, fgbg_two_step_controls_nimage]
            )

            compute_clusters_btn = gr.Button("Compute Cluster Correspondences", variant="secondary")

            cluster_overview_image = gr.Image(
                label="Cluster Overview",
                visible=False,
                height=400,
                type="pil"
            )

            # Cluster merging interface
            merge_panel_header = gr.Markdown("### Manual Cluster Merging", visible=False)
            with gr.Row(visible=False) as merge_panel_row:
                with gr.Column(scale=1):
                    cluster1_selector = gr.Dropdown(
                        label="Select First Cluster",
                        choices=[],
                        value=None,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    cluster2_selector = gr.Dropdown(
                        label="Select Second Cluster",
                        choices=[],
                        value=None,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    merge_button = gr.Button("Merge Clusters", variant="secondary")
            
            merge_preview_gallery = gr.Gallery(
                label="Clusters to Merge",
                visible=False,
                columns=2,
                height=300
            )

            generated_blend_image = gr.Image(
                label="Generated Blend",
                visible=False,
                type="pil"
            )

            weight_panel_header = gr.Markdown("### Cluster Controls", visible=False)

            cluster_accordions = []
            cluster_galleries = []
            cluster_slider_rows = []
            for cluster_idx in range(MAX_INTERP_CLUSTERS):
                with gr.Accordion(
                    f"Cluster {cluster_idx + 1}", open=False, visible=False
                ) as cluster_acc:
                    cluster_gallery = gr.Gallery(
                        label="Cluster Across Images",
                        height=400,
                        columns=4,
                        visible=False
                    )
                    slider_row = []
                    with gr.Row():
                        for image_idx in range(MAX_INTERP_IMAGES):
                            slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.0,
                                label=f"Image {image_idx + 1}",
                                visible=False
                            )
                            slider_row.append(slider)
                    cluster_accordions.append(cluster_acc)
                    cluster_galleries.append(cluster_gallery)
                    cluster_slider_rows.append(slider_row)

            cluster_slider_components = [slider for row in cluster_slider_rows for slider in row]

            def sync_base_choices(gallery_images):
                processed = load_gradio_images_helper(gallery_images)
                choices = [(f"Image {idx + 1}", idx) for idx in range(len(processed))]
                value = choices[0][1] if choices else None
                return gr.update(choices=choices, value=value)

            input_images.change(
                sync_base_choices,
                inputs=[input_images],
                outputs=[base_selector]
            )

            def _cluster_updates_placeholder(base_value):
                accordion_updates = [gr.update(visible=False) for _ in cluster_accordions]
                gallery_updates = [gr.update(visible=False, value=None) for _ in cluster_galleries]
                slider_updates = [gr.update(visible=False) for _ in cluster_slider_components]
                return (
                    {},
                    gr.update(value=None, visible=False),
                    [],
                    gr.update(value=None, visible=False),
                    gr.update(value=base_value),
                    gr.update(visible=False),
                    gr.update(visible=False),  # merge_panel_header
                    gr.update(visible=False),  # merge_panel_row
                    gr.update(choices=[], value=None),  # cluster1_selector
                    gr.update(choices=[], value=None),  # cluster2_selector
                    gr.update(visible=False),  # merge_preview_gallery
                    *accordion_updates,
                    *gallery_updates,
                    *slider_updates,
                )

            def compute_cluster_interface(
                gallery_images,
                base_idx,
                clustering_method_choice,
                n_clusters,
                match_method,
                n_superclusters,
                n_subclusters,
                supercluster_match,
                subcluster_match,
                n_fg_subclusters,
                n_bg_subclusters,
                fg_match,
                bg_match,
                current_config,
                current_weights,
            ):
                images = load_gradio_images_helper(gallery_images)
                if not images or len(images) < 2:
                    gr.Error("Please provide at least two training images.")
                    base_value = base_idx if base_idx is not None else None
                    return _cluster_updates_placeholder(base_value)

                if len(images) > MAX_INTERP_IMAGES:
                    gr.Error(f"This demo supports up to {MAX_INTERP_IMAGES} images for cluster interpolation.")
                    base_value = base_idx if base_idx is not None else None
                    return _cluster_updates_placeholder(base_value)

                # Determine which clustering method and effective number of clusters
                use_two_step = (clustering_method_choice == "Hierarchical 2-step")
                use_fgbg = (clustering_method_choice == "FG/BG 2-step")
                
                if use_fgbg:
                    effective_n_clusters = n_fg_subclusters + n_bg_subclusters
                elif use_two_step:
                    effective_n_clusters = n_superclusters * n_subclusters
                else:
                    effective_n_clusters = n_clusters
                
                if effective_n_clusters > MAX_INTERP_CLUSTERS:
                    gr.Error(f"Please select at most {MAX_INTERP_CLUSTERS} total clusters for this demo.")
                    base_value = base_idx if base_idx is not None else None
                    return _cluster_updates_placeholder(base_value)

                try:
                    base_idx_int = int(base_idx) if base_idx is not None else 0
                except (TypeError, ValueError):
                    base_idx_int = 0

                n_images = len(images)
                if base_idx_int < 0 or base_idx_int >= n_images:
                    gr.Info("Base selection reset to the first image.")
                    base_idx_int = 0

                images_tensor = torch.stack([dino_image_transform(image) for image in images])
                dino_embeds = extract_dino_features(images_tensor)
                
                # Perform clustering (single-step, hierarchical 2-step, or FG/BG 2-step)
                if use_fgbg:
                    supercluster_eigvecs, cluster_eigvecs, subcluster_to_supercluster, fg_indices = kway_cluster_per_image_two_step_fgbg(
                        dino_embeds,
                        n_foreground_subclusters=n_fg_subclusters,
                        n_background_subclusters=n_bg_subclusters
                    )
                elif use_two_step:
                    supercluster_eigvecs, cluster_eigvecs, subcluster_to_supercluster = kway_cluster_per_image_two_step(
                        dino_embeds,
                        n_superclusters=n_superclusters,
                        n_subclusters_per_supercluster=n_subclusters
                    )
                else:
                    cluster_eigvecs = kway_cluster_per_image(dino_embeds, n_clusters=n_clusters, gamma=None)
                
                joint_eigvecs, joint_colors = ncut_tsne_multiple_images(dino_embeds, n_eig=30, gamma=None)
                discrete_colors = get_discrete_colors_from_clusters(joint_colors, cluster_eigvecs)

                cluster_mappings: List[np.ndarray] = []
                for image_idx in range(n_images):
                    if image_idx == base_idx_int:
                        cluster_mappings.append(np.arange(effective_n_clusters))
                    else:
                        if use_fgbg:
                            mapping = match_centers_two_step_fgbg(
                                dino_embeds[base_idx_int],
                                dino_embeds[image_idx],
                                cluster_eigvecs[base_idx_int],
                                cluster_eigvecs[image_idx],
                                subcluster_to_supercluster[base_idx_int],
                                subcluster_to_supercluster[image_idx],
                                n_bg_subclusters,
                                n_fg_subclusters,
                                background_match_method=bg_match,
                                foreground_match_method=fg_match,
                            )
                        elif use_two_step:
                            mapping = match_centers_two_step(
                                dino_embeds[base_idx_int],
                                dino_embeds[image_idx],
                                supercluster_eigvecs[base_idx_int],
                                supercluster_eigvecs[image_idx],
                                cluster_eigvecs[base_idx_int],
                                cluster_eigvecs[image_idx],
                                subcluster_to_supercluster[base_idx_int],
                                subcluster_to_supercluster[image_idx],
                                supercluster_match_method=supercluster_match,
                                subcluster_match_method=subcluster_match,
                            )
                        else:
                            mapping = match_centers_two_images(
                                dino_embeds[base_idx_int],
                                dino_embeds[image_idx],
                                cluster_eigvecs[base_idx_int],
                                cluster_eigvecs[image_idx],
                                match_method=match_method,
                            )
                        cluster_mappings.append(mapping)

                cluster_orders = []
                for image_idx in range(n_images):
                    if image_idx == base_idx_int:
                        cluster_orders.append(np.arange(effective_n_clusters))
                    else:
                        cluster_orders.append(cluster_mappings[image_idx])

                overview_image = get_correspondence_plot(
                    images_tensor,
                    cluster_eigvecs,
                    cluster_orders,
                    discrete_colors,
                    hw=16 * 2,
                    n_cols=10,
                )

                token_count = cluster_eigvecs.shape[1]
                spatial_tokens = max(token_count - 1, 1)
                hw = max(int(round(np.sqrt(spatial_tokens))), 1)

                def masked_cluster_preview(image_tensor, eigenvectors, cluster_id):
                    base_img = image_inverse_transform(image_tensor.cpu()).resize(
                        (196, 196), resample=Image.Resampling.LANCZOS
                    )
                    labels = eigenvectors.argmax(-1).cpu().numpy()
                    patch_mask = (labels[1:] == cluster_id).astype(np.uint8).reshape(hw, hw)
                    mask_img = Image.fromarray(patch_mask * 255).resize(
                        base_img.size, resample=Image.Resampling.NEAREST
                    )
                    mask_arr = np.array(mask_img).astype(np.float32) / 255.0
                    mask_arr = np.expand_dims(mask_arr, axis=-1)
                    base_arr = np.array(base_img).astype(np.float32) / 255.0
                    overlay = base_arr * (0.2 + 0.8 * mask_arr)
                    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
                    return Image.fromarray(overlay)

                cluster_previews: List[List[Image.Image]] = []
                for cluster_idx in range(effective_n_clusters):
                    preview_images = []
                    for image_idx in range(n_images):
                        mapped_cluster = (
                            cluster_idx if image_idx == base_idx_int else int(cluster_mappings[image_idx][cluster_idx])
                        )
                        preview_images.append(
                            masked_cluster_preview(
                                images_tensor[image_idx],
                                cluster_eigvecs[image_idx],
                                mapped_cluster,
                            )
                        )
                    cluster_previews.append(preview_images)

                default_weights = [
                    [1.0 if image_idx == base_idx_int else 0.0 for image_idx in range(n_images)]
                    for _ in range(effective_n_clusters)
                ]

                config_payload = {
                    "n_images": n_images,
                    "n_clusters": effective_n_clusters,
                    "base_idx": base_idx_int,
                    "match_method": match_method,
                    "clustering_method": clustering_method_choice,
                    "use_two_step": use_two_step,
                    "n_superclusters": n_superclusters if use_two_step else None,
                    "n_subclusters": n_subclusters if use_two_step else None,
                    "supercluster_match": supercluster_match if use_two_step else None,
                    "subcluster_match": subcluster_match if use_two_step else None,
                    "use_fgbg": use_fgbg,
                    "n_fg_subclusters": n_fg_subclusters if use_fgbg else None,
                    "n_bg_subclusters": n_bg_subclusters if use_fgbg else None,
                    "fg_match": fg_match if use_fgbg else None,
                    "bg_match": bg_match if use_fgbg else None,
                    "dino_embeds": dino_embeds,
                    "cluster_eigvecs": cluster_eigvecs,
                    "cluster_mappings": cluster_mappings,
                    "cluster_previews": cluster_previews,
                }

                accordion_updates = [
                    gr.update(visible=(idx < effective_n_clusters))
                    for idx in range(len(cluster_accordions))
                ]

                gallery_updates = []
                for idx in range(len(cluster_galleries)):
                    if idx < effective_n_clusters:
                        gallery_updates.append(
                            gr.update(
                                value=cluster_previews[idx],
                                visible=True,
                                columns=n_images,
                            )
                        )
                    else:
                        gallery_updates.append(gr.update(visible=False, value=None))

                slider_updates = []
                for cluster_idx in range(len(cluster_accordions)):
                    for image_idx in range(MAX_INTERP_IMAGES):
                        if cluster_idx < effective_n_clusters and image_idx < n_images:
                            label = f"Image {image_idx + 1}"
                            if image_idx == base_idx_int:
                                label += " (base)"
                            slider_updates.append(
                                gr.update(
                                    visible=True,
                                    value=default_weights[cluster_idx][image_idx],
                                    label=label,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                )
                            )
                        else:
                            slider_updates.append(gr.update(visible=False))

                weight_panel_visibility = gr.update(visible=True)

                base_choices = [(f"Image {idx + 1}", idx) for idx in range(n_images)]
                base_dropdown_update = gr.update(
                    choices=base_choices,
                    value=base_idx_int,
                )

                # Update merge panel
                cluster_choices = [(f"Cluster {i + 1}", i) for i in range(effective_n_clusters)]
                merge_panel_header_update = gr.update(visible=True)
                merge_panel_row_update = gr.update(visible=True)
                cluster1_update = gr.update(choices=cluster_choices, value=None)
                cluster2_update = gr.update(choices=cluster_choices, value=None)
                merge_preview_update = gr.update(visible=False)

                return (
                    config_payload,
                    gr.update(value=overview_image, visible=True),
                    default_weights,
                    gr.update(value=None, visible=False),
                    base_dropdown_update,
                    weight_panel_visibility,
                    merge_panel_header_update,
                    merge_panel_row_update,
                    cluster1_update,
                    cluster2_update,
                    merge_preview_update,
                    *accordion_updates,
                    *gallery_updates,
                    *slider_updates,
                )

            compute_clusters_btn.click(
                compute_cluster_interface,
                inputs=[
                    input_images,
                    base_selector,
                    clustering_method_nimage,
                    cluster_count_slider,
                    match_method_dropdown,
                    n_superclusters_nimage,
                    n_subclusters_per_supercluster_nimage,
                    supercluster_match_method_nimage,
                    subcluster_match_method_nimage,
                    n_foreground_subclusters_nimage,
                    n_background_subclusters_nimage,
                    foreground_match_method_nimage,
                    background_match_method_nimage,
                    cluster_config_state,
                    cluster_weight_state,
                ],
                outputs=[
                    cluster_config_state,
                    cluster_overview_image,
                    cluster_weight_state,
                    generated_blend_image,
                    base_selector,
                    weight_panel_header,
                    merge_panel_header,
                    merge_panel_row,
                    cluster1_selector,
                    cluster2_selector,
                    merge_preview_gallery,
                    *cluster_accordions,
                    *cluster_galleries,
                    *cluster_slider_components,
                ],
            )

            # Handler for updating merge preview when clusters are selected
            def update_merge_preview(cluster1_idx, cluster2_idx, config):
                if cluster1_idx is None or cluster2_idx is None:
                    return gr.update(visible=False, value=None)
                
                if cluster1_idx == cluster2_idx:
                    gr.Warning("Please select two different clusters to merge.")
                    return gr.update(visible=False, value=None)
                
                if not config or "cluster_previews" not in config:
                    return gr.update(visible=False, value=None)
                
                cluster_previews = config["cluster_previews"]
                n_clusters = config.get("n_clusters", 0)
                
                if cluster1_idx >= n_clusters or cluster2_idx >= n_clusters:
                    return gr.update(visible=False, value=None)
                
                # Show preview of both clusters
                preview_images = cluster_previews[cluster1_idx] + cluster_previews[cluster2_idx]
                return gr.update(visible=True, value=preview_images, columns=len(cluster_previews[cluster1_idx]))
            
            cluster1_selector.change(
                update_merge_preview,
                inputs=[cluster1_selector, cluster2_selector, cluster_config_state],
                outputs=[merge_preview_gallery]
            )
            
            cluster2_selector.change(
                update_merge_preview,
                inputs=[cluster1_selector, cluster2_selector, cluster_config_state],
                outputs=[merge_preview_gallery]
            )

            # Handler for merging clusters
            def merge_clusters_handler(
                gallery_images,
                cluster1_idx,
                cluster2_idx,
                config,
                current_weights
            ):
                if cluster1_idx is None or cluster2_idx is None:
                    gr.Error("Please select two clusters to merge.")
                    return _cluster_updates_placeholder(config.get("base_idx", 0))
                
                if cluster1_idx == cluster2_idx:
                    gr.Error("Please select two different clusters to merge.")
                    return _cluster_updates_placeholder(config.get("base_idx", 0))
                
                if not config:
                    gr.Error("Please compute clusters first.")
                    return _cluster_updates_placeholder(None)
                
                # Get configuration
                n_images = config.get("n_images")
                n_clusters = config.get("n_clusters")
                base_idx = config.get("base_idx", 0)
                match_method = config.get("match_method", "hungarian")
                dino_embeds = config.get("dino_embeds")
                
                # Merge: keep cluster1, remove cluster2
                # Simple union - reassign cluster2 points to cluster1, renumber remaining clusters
                cluster_eigvecs = config.get("cluster_eigvecs")
                cluster_mappings = config.get("cluster_mappings")
                
                # Create new cluster eigenvectors with merged clusters
                new_cluster_eigvecs = []
                new_n_clusters = n_clusters - 1
                
                # Create a mapping from old cluster indices to new cluster indices
                # cluster1_idx stays the same, cluster2_idx is removed, indices after cluster2 shift down
                def get_new_cluster_idx(old_idx, c1_idx, c2_idx):
                    """Map old cluster index to new cluster index after merge."""
                    if old_idx == c2_idx:
                        return c1_idx  # cluster2 merges into cluster1
                    elif old_idx > c2_idx:
                        return old_idx - 1  # shift down indices after cluster2
                    else:
                        return old_idx  # keep same index if before cluster2
                
                for img_idx in range(n_images):
                    eigvec = cluster_eigvecs[img_idx].clone()
                    old_labels = eigvec.argmax(-1).clone()
                    
                    # For each image, find which clusters map to cluster1 and cluster2 in base image
                    if img_idx == base_idx:
                        # Direct merge in base image: union of cluster1 and cluster2
                        new_labels = old_labels.clone()
                        
                        # Simple reassignment: cluster2 -> cluster1, then renumber
                        for token_idx in range(len(new_labels)):
                            old_cluster = old_labels[token_idx].item()
                            new_labels[token_idx] = get_new_cluster_idx(old_cluster, cluster1_idx, cluster2_idx)
                    else:
                        # For other images, use mapping to find corresponding clusters
                        mapping = cluster_mappings[img_idx]
                        
                        # Find which clusters in this image map to cluster1 and cluster2 in base
                        # mapping[base_cluster_idx] = image_cluster_idx
                        mapped_c1 = int(mapping[cluster1_idx])
                        mapped_c2 = int(mapping[cluster2_idx])
                        
                        # Create inverse mapping: which base cluster does each image cluster map to?
                        # We need to know which base cluster each image cluster corresponds to
                        inverse_mapping = np.full(n_clusters, -1, dtype=np.int64)
                        for base_c_idx, img_c_idx in enumerate(mapping):
                            if img_c_idx < n_clusters:
                                inverse_mapping[img_c_idx] = base_c_idx
                        
                        # Now merge in image space and renumber based on base cluster assignments
                        new_labels = old_labels.clone()
                        for token_idx in range(len(old_labels)):
                            old_img_cluster = old_labels[token_idx].item()
                            
                            # Find which base cluster this image cluster maps to
                            base_cluster = inverse_mapping[old_img_cluster]
                            
                            if base_cluster >= 0:
                                # Apply the merge in base space, then get new index
                                new_base_cluster = get_new_cluster_idx(base_cluster, cluster1_idx, cluster2_idx)
                                new_labels[token_idx] = new_base_cluster
                            else:
                                # Shouldn't happen, but keep the label
                                new_labels[token_idx] = old_img_cluster
                    
                    # Convert labels back to eigenvectors (one-hot)
                    new_eigvec = torch.zeros(eigvec.shape[0], new_n_clusters, device=eigvec.device)
                    for i in range(new_n_clusters):
                        new_eigvec[new_labels == i, i] = 1.0
                    
                    new_cluster_eigvecs.append(new_eigvec)
                
                new_cluster_eigvecs = torch.stack(new_cluster_eigvecs)
                
                # Create new cluster mappings based on the merge
                # Each image's cluster i maps to base cluster i now (identity mapping)
                # because we've already applied the merge in each image's label space
                new_cluster_mappings: List[np.ndarray] = []
                for image_idx in range(n_images):
                    new_cluster_mappings.append(np.arange(new_n_clusters))
                
                # Generate new overview image
                images = load_gradio_images_helper(gallery_images)
                images_tensor = torch.stack([dino_image_transform(image) for image in images])
                joint_eigvecs, joint_colors = ncut_tsne_multiple_images(dino_embeds, n_eig=30, gamma=None)
                discrete_colors = get_discrete_colors_from_clusters(joint_colors, new_cluster_eigvecs)
                
                cluster_orders = []
                for image_idx in range(n_images):
                    if image_idx == base_idx:
                        cluster_orders.append(np.arange(new_n_clusters))
                    else:
                        cluster_orders.append(new_cluster_mappings[image_idx])
                
                overview_image = get_correspondence_plot(
                    images_tensor,
                    new_cluster_eigvecs,
                    cluster_orders,
                    discrete_colors,
                    hw=16 * 2,
                    n_cols=10,
                )
                
                # Generate cluster previews
                token_count = new_cluster_eigvecs.shape[1]
                spatial_tokens = max(token_count - 1, 1)
                hw = max(int(round(np.sqrt(spatial_tokens))), 1)
                
                def masked_cluster_preview(image_tensor, eigenvectors, cluster_id):
                    base_img = image_inverse_transform(image_tensor.cpu()).resize(
                        (196, 196), resample=Image.Resampling.LANCZOS
                    )
                    labels = eigenvectors.argmax(-1).cpu().numpy()
                    patch_mask = (labels[1:] == cluster_id).astype(np.uint8).reshape(hw, hw)
                    mask_img = Image.fromarray(patch_mask * 255).resize(
                        base_img.size, resample=Image.Resampling.NEAREST
                    )
                    mask_arr = np.array(mask_img).astype(np.float32) / 255.0
                    mask_arr = np.expand_dims(mask_arr, axis=-1)
                    base_arr = np.array(base_img).astype(np.float32) / 255.0
                    overlay = base_arr * (0.2 + 0.8 * mask_arr)
                    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
                    return Image.fromarray(overlay)
                
                cluster_previews: List[List[Image.Image]] = []
                for cluster_idx in range(new_n_clusters):
                    preview_images = []
                    for image_idx in range(n_images):
                        mapped_cluster = (
                            cluster_idx if image_idx == base_idx else int(new_cluster_mappings[image_idx][cluster_idx])
                        )
                        preview_images.append(
                            masked_cluster_preview(
                                images_tensor[image_idx],
                                new_cluster_eigvecs[image_idx],
                                mapped_cluster,
                            )
                        )
                    cluster_previews.append(preview_images)
                
                # Merge weights for the merged cluster
                new_weights = []
                for cluster_idx in range(n_clusters):
                    if cluster_idx == cluster1_idx:
                        # Average weights from both clusters
                        if current_weights and len(current_weights) > cluster1_idx and len(current_weights) > cluster2_idx:
                            merged_weight = [
                                (current_weights[cluster1_idx][i] + current_weights[cluster2_idx][i]) / 2.0
                                for i in range(n_images)
                            ]
                        else:
                            merged_weight = [1.0 if i == base_idx else 0.0 for i in range(n_images)]
                        new_weights.append(merged_weight)
                    elif cluster_idx == cluster2_idx:
                        # Skip cluster2
                        continue
                    else:
                        # Keep other clusters' weights
                        if current_weights and len(current_weights) > cluster_idx:
                            new_weights.append(current_weights[cluster_idx][:n_images])
                        else:
                            new_weights.append([1.0 if i == base_idx else 0.0 for i in range(n_images)])
                
                # Update config
                new_config = {
                    "n_images": n_images,
                    "n_clusters": new_n_clusters,
                    "base_idx": base_idx,
                    "match_method": match_method,
                    "dino_embeds": dino_embeds,
                    "cluster_eigvecs": new_cluster_eigvecs,
                    "cluster_mappings": new_cluster_mappings,
                    "cluster_previews": cluster_previews,
                }
                
                # Update UI components
                accordion_updates = [
                    gr.update(visible=(idx < new_n_clusters), label=f"Cluster {idx + 1}")
                    for idx in range(len(cluster_accordions))
                ]
                
                gallery_updates = []
                for idx in range(len(cluster_galleries)):
                    if idx < new_n_clusters:
                        gallery_updates.append(
                            gr.update(
                                value=cluster_previews[idx],
                                visible=True,
                                columns=n_images,
                            )
                        )
                    else:
                        gallery_updates.append(gr.update(visible=False, value=None))
                
                slider_updates = []
                for cluster_idx in range(len(cluster_accordions)):
                    for image_idx in range(MAX_INTERP_IMAGES):
                        if cluster_idx < new_n_clusters and image_idx < n_images:
                            label = f"Image {image_idx + 1}"
                            if image_idx == base_idx:
                                label += " (base)"
                            slider_updates.append(
                                gr.update(
                                    visible=True,
                                    value=new_weights[cluster_idx][image_idx],
                                    label=label,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                )
                            )
                        else:
                            slider_updates.append(gr.update(visible=False))
                
                # Update merge panel
                cluster_choices = [(f"Cluster {i + 1}", i) for i in range(new_n_clusters)]
                merge_panel_header_update = gr.update(visible=True)
                merge_panel_row_update = gr.update(visible=True)
                cluster1_update = gr.update(choices=cluster_choices, value=None)
                cluster2_update = gr.update(choices=cluster_choices, value=None)
                merge_preview_update = gr.update(visible=False)
                
                gr.Info(f"Merged Cluster {cluster2_idx + 1} into Cluster {cluster1_idx + 1}. Now {new_n_clusters} clusters.")
                
                return (
                    new_config,
                    gr.update(value=overview_image, visible=True),
                    new_weights,
                    gr.update(value=None, visible=False),
                    gr.update(value=base_idx),
                    gr.update(visible=True),
                    merge_panel_header_update,
                    merge_panel_row_update,
                    cluster1_update,
                    cluster2_update,
                    merge_preview_update,
                    *accordion_updates,
                    *gallery_updates,
                    *slider_updates,
                )
            
            merge_button.click(
                merge_clusters_handler,
                inputs=[
                    input_images,
                    cluster1_selector,
                    cluster2_selector,
                    cluster_config_state,
                    cluster_weight_state,
                ],
                outputs=[
                    cluster_config_state,
                    cluster_overview_image,
                    cluster_weight_state,
                    generated_blend_image,
                    base_selector,
                    weight_panel_header,
                    merge_panel_header,
                    merge_panel_row,
                    cluster1_selector,
                    cluster2_selector,
                    merge_preview_gallery,
                    *cluster_accordions,
                    *cluster_galleries,
                    *cluster_slider_components,
                ],
            )

            def make_slider_handler(cluster_idx: int, image_idx: int):
                def _update_slider(value, weights):
                    if not weights or cluster_idx >= len(weights):
                        return weights
                    if image_idx >= len(weights[cluster_idx]):
                        return weights
                    updated = copy.deepcopy(weights)
                    updated[cluster_idx][image_idx] = float(value)
                    return updated

                return _update_slider

            for cluster_idx, slider_row in enumerate(cluster_slider_rows):
                for image_idx, slider in enumerate(slider_row):
                    slider.change(
                        make_slider_handler(cluster_idx, image_idx),
                        inputs=[slider, cluster_weight_state],
                        outputs=[cluster_weight_state],
                    )

            def generate_cluster_image(gallery_images, model, config, weights):
                if model is None or model == []:
                    gr.Error("Please train a model first.")
                    return gr.update(value=None, visible=False)

                if not config or not weights:
                    gr.Error("Compute clusters before generating a blend.")
                    return gr.update(value=None, visible=False)

                n_clusters = config.get("n_clusters")
                n_images = config.get("n_images")
                base_idx = config.get("base_idx", 0)
                match_method = config.get("match_method", "hungarian")
                
                # Extract clustering method and parameters
                clustering_method = config.get("clustering_method", "Single-step")
                use_two_step = (clustering_method == "Hierarchical 2-step")
                use_fgbg = (clustering_method == "FG/BG 2-step")
                
                # Hierarchical 2-step parameters
                n_superclusters = config.get("n_superclusters")
                n_subclusters = config.get("n_subclusters")
                supercluster_match = config.get("supercluster_match", "hungarian")
                subcluster_match = config.get("subcluster_match", "hungarian")
                
                # FG/BG 2-step parameters
                n_fg_subclusters = config.get("n_fg_subclusters")
                n_bg_subclusters = config.get("n_bg_subclusters")
                fg_match = config.get("fg_match", "hungarian")
                bg_match = config.get("bg_match", "hungarian")
                
                # Extract precomputed cluster data
                precomputed_dino_embeds = config.get("dino_embeds")
                precomputed_cluster_eigvecs = config.get("cluster_eigvecs")
                precomputed_cluster_mappings = config.get("cluster_mappings")

                images = load_gradio_images_helper(gallery_images)
                if len(images) != n_images:
                    gr.Error("Training images changed. Re-compute clusters to continue.")
                    return gr.update(value=None, visible=False)

                if len(weights) < n_clusters:
                    gr.Error("Weight configuration incomplete. Recompute clusters.")
                    return gr.update(value=None, visible=False)

                trimmed_weights: List[List[float]] = []
                for cluster_idx in range(n_clusters):
                    weight_row = weights[cluster_idx][:n_images]
                    if len(weight_row) != n_images:
                        gr.Error("Weight configuration mismatch. Recompute clusters.")
                        return gr.update(value=None, visible=False)
                    trimmed_weights.append([float(w) for w in weight_row])

                try:
                    generated = perform_n_image_interpolation_per_cluster(
                        images,
                        base_idx,
                        model,
                        trimmed_weights,
                        n_clusters=n_clusters,
                        match_method=match_method,
                        use_dino_matching=True,
                        use_two_step_clustering=use_two_step,
                        n_superclusters=n_superclusters if use_two_step else 3,
                        n_subclusters_per_supercluster=n_subclusters if use_two_step else 5,
                        supercluster_match_method=supercluster_match,
                        subcluster_match_method=subcluster_match,
                        use_two_step_fgbg_clustering=use_fgbg,
                        n_foreground_subclusters=n_fg_subclusters if use_fgbg else 10,
                        n_background_subclusters=n_bg_subclusters if use_fgbg else 5,
                        foreground_match_method=fg_match,
                        background_match_method=bg_match,
                        seed=None,
                        config_path=DEFAULT_CONFIG_PATH,
                        precomputed_dino_embeds=precomputed_dino_embeds,
                        precomputed_cluster_eigenvectors=precomputed_cluster_eigvecs,
                        precomputed_cluster_mappings=precomputed_cluster_mappings,
                    )
                except Exception as exc:
                    gr.Error(f"Generation failed: {exc}")
                    return gr.update(value=None, visible=False)

                if not generated:
                    gr.Error("No image was generated. Try adjusting the weights.")
                    return gr.update(value=None, visible=False)

                return gr.update(value=generated[0], visible=True)

            generate_button = gr.Button("Generate Blend", variant="primary")
            generate_button.click(
                generate_cluster_image,
                inputs=[input_images, model_state, cluster_config_state, cluster_weight_state],
                outputs=[generated_blend_image],
            )

            # Add cluster weight visualization
            gr.Markdown("### Cluster Weight Visualization")
            gr.Markdown("""
            This visualization shows how clusters are weighted across images. Each image is shown with 
            a colored transparent overlay for all clusters. Higher weight means more transparency (less visible), 
            making it easy to see which regions are contributing least to the final blend.
            """)
            
            with gr.Row():
                visualize_weights_button = gr.Button("Visualize Cluster Weights", variant="secondary")
                show_overlay_only = gr.Checkbox(
                    label="Show overlay only (no base image)",
                    value=False,
                    interactive=True
                )
            
            weight_visualization_gallery = gr.Gallery(
                label="Cluster Weight Visualization",
                visible=False,
                columns=None,
                height=400
            )
            
            def visualize_cluster_weights(gallery_images, config, weights, overlay_only):
                """Generate visualization of cluster weights as colored transparent overlays."""
                if not config or not weights:
                    gr.Error("Compute clusters before visualizing weights.")
                    return gr.update(value=None, visible=False)
                
                n_clusters = config.get("n_clusters")
                n_images = config.get("n_images")
                cluster_eigvecs = config.get("cluster_eigvecs")
                
                images = load_gradio_images_helper(gallery_images)
                if len(images) != n_images:
                    gr.Error("Training images changed. Re-compute clusters to continue.")
                    return gr.update(value=None, visible=False)
                
                # Generate a unique color for each cluster
                # Use HSV color space to ensure visually distinct colors
                cluster_colors = []
                for i in range(n_clusters):
                    hue = i / n_clusters
                    # Convert HSV to RGB (S=1, V=1 for maximum saturation and brightness)
                    import colorsys
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    cluster_colors.append((int(r * 255), int(g * 255), int(b * 255)))
                
                # Create visualization for each image
                visualizations = []
                
                # Calculate spatial dimensions from eigenvectors
                token_count = cluster_eigvecs.shape[1]
                spatial_tokens = max(token_count - 1, 1)
                hw = max(int(round(np.sqrt(spatial_tokens))), 1)
                
                for img_idx in range(n_images):
                    # Get base image and resize
                    base_img = images[img_idx].resize((256, 256), resample=Image.Resampling.LANCZOS)
                    base_arr = np.array(base_img).astype(np.float32)
                    
                    # Get cluster labels for this image (excluding CLS token)
                    labels = cluster_eigvecs[img_idx].argmax(-1).cpu().numpy()[1:]  # Skip CLS token
                    labels = labels.reshape(hw, hw)
                    
                    if overlay_only:
                        # For overlay-only mode: create a composite where each cluster
                        # goes from white (w=0) to full color (w=1)
                        # Start with white background
                        composite = np.ones((hw, hw, 3), dtype=np.float32) * 255.0
                        
                        for cluster_idx in range(n_clusters):
                            # Get weight for this cluster and image
                            if cluster_idx < len(weights) and img_idx < len(weights[cluster_idx]):
                                weight = weights[cluster_idx][img_idx]
                            else:
                                weight = 0.0
                            
                            # Create mask for this cluster
                            cluster_mask = (labels == cluster_idx).astype(np.float32)
                            
                            # For overlay-only: higher weight = darker/more opaque color
                            # Interpolate from white (255, 255, 255) at w=0 to cluster color at w=1
                            color = np.array(cluster_colors[cluster_idx], dtype=np.float32)
                            
                            # Linear interpolation: result = white * (1 - weight) + color * weight
                            for c in range(3):
                                cluster_contribution = 255.0 * (1.0 - weight) + color[c] * weight
                                # Apply to masked regions, keeping other regions as they are
                                composite[:, :, c] = np.where(
                                    cluster_mask > 0,
                                    cluster_contribution,
                                    composite[:, :, c]
                                )
                        
                        # Resize composite to match display size
                        composite_resized = np.zeros((256, 256, 3), dtype=np.float32)
                        for c in range(3):
                            comp_img = Image.fromarray(composite[:, :, c].astype(np.uint8))
                            comp_img = comp_img.resize((256, 256), resample=Image.Resampling.NEAREST)
                            composite_resized[:, :, c] = np.array(comp_img)
                        
                        composite_resized = np.clip(composite_resized, 0, 255).astype(np.uint8)
                        result_img = Image.fromarray(composite_resized)
                    else:
                        # For blended mode: create colored overlay as before
                        # Create colored overlay combining all clusters
                        overlay = np.zeros((hw, hw, 3), dtype=np.float32)
                        alpha_map = np.zeros((hw, hw), dtype=np.float32)
                        
                        for cluster_idx in range(n_clusters):
                            # Get weight for this cluster and image
                            if cluster_idx < len(weights) and img_idx < len(weights[cluster_idx]):
                                weight = weights[cluster_idx][img_idx]
                            else:
                                weight = 0.0
                            
                            # Create mask for this cluster
                            cluster_mask = (labels == cluster_idx).astype(np.float32)
                            
                            # Invert weight: higher weight = more transparent
                            # Map weight [0, 1] to opacity [1, 0]
                            opacity = 1.0 - weight
                            
                            # Add this cluster's colored contribution
                            color = np.array(cluster_colors[cluster_idx], dtype=np.float32)
                            for c in range(3):
                                overlay[:, :, c] += cluster_mask * color[c] * opacity
                            
                            # Track cumulative alpha
                            alpha_map += cluster_mask * opacity
                        
                        # Normalize alpha map to [0, 1] and ensure it doesn't exceed 1
                        alpha_map = np.clip(alpha_map, 0, 1)
                        
                        # Resize overlay and alpha to match base image
                        overlay_resized = np.zeros((256, 256, 3), dtype=np.float32)
                        alpha_resized = np.zeros((256, 256), dtype=np.float32)
                        
                        for c in range(3):
                            overlay_img = Image.fromarray(overlay[:, :, c].astype(np.uint8))
                            overlay_img = overlay_img.resize((256, 256), resample=Image.Resampling.NEAREST)
                            overlay_resized[:, :, c] = np.array(overlay_img)
                        
                        alpha_img = Image.fromarray((alpha_map * 255).astype(np.uint8))
                        alpha_img = alpha_img.resize((256, 256), resample=Image.Resampling.NEAREST)
                        alpha_resized = np.array(alpha_img).astype(np.float32) / 255.0
                        
                        # Blend base image with colored overlay
                        alpha_resized = np.expand_dims(alpha_resized, axis=-1)
                        transparency_factor = 1.0  # Adjust this factor to control overlay visibility
                        blended = base_arr * (1 - alpha_resized * transparency_factor) + overlay_resized * alpha_resized * transparency_factor
                        blended = np.clip(blended, 0, 255).astype(np.uint8)
                        result_img = Image.fromarray(blended)
                    
                    visualizations.append(result_img)
                
                return gr.update(value=visualizations, visible=True, columns=n_images)
            
            visualize_weights_button.click(
                visualize_cluster_weights,
                inputs=[input_images, cluster_config_state, cluster_weight_state, show_overlay_only],
                outputs=[weight_visualization_gallery],
            )

    return demo


# ===== Main Application Entry Point =====

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
