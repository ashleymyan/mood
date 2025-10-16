"""
DINO Correspondence Analysis Module

This module provides functions for analyzing visual correspondences between images
using DINO features, normalized cuts (NCut), and clustering techniques.
"""

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from einops import rearrange

from extract_features import img_transform_inv
from ipadapter_model import image_grid
from ncut_pytorch import ncut_fn, kway_ncut, convert_to_lab_color
from ncut_pytorch.color import tsne_color
from ncut_pytorch.utils.gamma import find_gamma_by_degree


# ===== Core NCut and Clustering Functions =====

def ncut_tsne_multiple_images(image_embeds, n_eig=50, gamma=0.5, degree=0.5):
    """
    Apply NCut and t-SNE coloring to multiple image embeddings.
    
    image_embeds is (batch, length, channels)
    """
    batch_size, length, channels = image_embeds.shape
    flattened_input = image_embeds.flatten(end_dim=-2)
    
    if gamma is None:
        gamma = find_gamma_by_degree(flattened_input, degree)
    
    eigenvectors, eigenvalues = ncut_fn(
        flattened_input, n_eig=n_eig, gamma=gamma, device='cuda'
    )
    
    rgb_colors = tsne_color(eigenvectors, n_dim=3, device='cuda', perplexity=50)
    rgb_colors = convert_to_lab_color(rgb_colors)
    
    # Reshape back to original batch structure
    rgb_colors = rearrange(rgb_colors, '(b l) c -> b l c', b=batch_size)
    eigenvectors = rearrange(eigenvectors, '(b l) c -> b l c', b=batch_size)
    
    return eigenvectors, rgb_colors


def _kway_cluster_single_image(image_embeds, n_clusters, gamma=0.5, degree=0.5):
    length, channels = image_embeds.shape
    flattened_input = image_embeds.flatten(end_dim=-2)
    
    if gamma is None:
        gamma = find_gamma_by_degree(flattened_input, degree)
    else:
        gamma = gamma * image_embeds.var(0).sum().item()
    
    # Calculate number of eigenvectors needed
    n_eig = min(n_clusters * 2 + 6, flattened_input.shape[0] // 2 - 1)
    
    eigenvectors, _ = ncut_fn(
        flattened_input, n_eig=n_eig, gamma=gamma, device='cuda'
    )
    
    continuous_clusters = kway_ncut(eigenvectors[:, :n_clusters])
    return continuous_clusters


def kway_cluster_per_image(image_embeds, n_clusters, gamma=None, degree=0.5):
    """
    Perform k-way clustering on each image separately.
    
    image_embeds is (length, channels)
    return (length, clusters)
    """
    clustered_eigenvectors = []
    
    for i in range(image_embeds.shape[0]):
        eigenvector = _kway_cluster_single_image(
            image_embeds[i], n_clusters, gamma, degree
        )
        clustered_eigenvectors.append(eigenvector)
    
    return torch.stack(clustered_eigenvectors)


def kway_cluster_multiple_images(image_embeds, n_clusters, gamma=0.5, degree=0.5):
    """
    Perform k-way clustering on multiple images jointly.
    
    image_embeds is (batch, length, channels)
    return (batch, length, clusters)
    """
    batch_size, length, channels = image_embeds.shape
    flattened_input = image_embeds.flatten(end_dim=-2)
    
    if gamma is None:
        gamma = find_gamma_by_degree(flattened_input, degree)
    
    # Calculate number of eigenvectors needed
    n_eig = min(n_clusters * 2 + 6, flattened_input.shape[0] // 2 - 1)
    
    eigenvectors, _ = ncut_fn(
        flattened_input, n_eig=n_eig, gamma=gamma, device='cuda'
    )
    
    continuous_clusters = kway_ncut(eigenvectors[:, :n_clusters])
    continuous_clusters = rearrange(
        continuous_clusters, '(b l) c -> b l c', b=batch_size
    )
    
    return continuous_clusters


# ===== Color and Visualization Functions =====

def get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors):

    n_clusters = cluster_eigenvectors.shape[-1]
    discrete_colors = np.zeros_like(joint_colors)
    
    for img_idx in range(joint_colors.shape[0]):
        colors = joint_colors[img_idx]
        eigenvector = cluster_eigenvectors[img_idx].cpu().numpy()
        cluster_labels = eigenvector.argmax(-1)
        discrete_img_colors = np.zeros_like(colors)
        
        for cluster_idx in range(n_clusters):
            cluster_mask = cluster_labels == cluster_idx
            if cluster_mask.sum() > 0:
                # Use mean color for each cluster
                discrete_img_colors[cluster_mask] = colors[cluster_mask].mean(0)
        
        discrete_colors[img_idx] = discrete_img_colors
    
    # Convert to uint8 format
    discrete_colors = (discrete_colors * 255).astype(np.uint8)
    return discrete_colors


# ===== Center Matching Functions =====

def get_cluster_center_features(image_embeds, cluster_labels, n_clusters):

    center_features = torch.zeros((n_clusters, image_embeds.shape[-1]))
    
    for cluster_idx in range(n_clusters):
        cluster_mask = cluster_labels == cluster_idx
        
        if cluster_mask.sum() > 0:
            center_features[cluster_idx] = image_embeds[cluster_mask].mean(0)
        else:
            # Use a unique identifier for empty clusters
            center_features[cluster_idx] = torch.ones_like(image_embeds[0]) * 114514
    
    return center_features


def cosine_similarity(matrix_a, matrix_b):
    normalized_a = matrix_a / matrix_a.norm(dim=-1, keepdim=True)
    normalized_b = matrix_b / matrix_b.norm(dim=-1, keepdim=True)
    return normalized_a @ normalized_b.T


def hungarian_match_centers(center_features1, center_features2):
    distances = torch.cdist(center_features1, center_features2)
    distances = distances.cpu().detach().numpy()
    
    _, column_indices = linear_sum_assignment(distances)
    return column_indices


def argmin_matching(center_features1, center_features2):
    distances = torch.cdist(center_features1, center_features2)
    distances = distances.cpu().detach().numpy()
    return np.argmin(distances, axis=-1)


def match_cluster_centers(image_embed1, image_embed2, eigvec1, eigvec2, 
                         match_method='hungarian'):
    cluster_labels1 = eigvec1.argmax(-1).cpu().numpy()
    cluster_labels2 = eigvec2.argmax(-1).cpu().numpy()
    n_clusters = eigvec1.shape[-1]
    
    center_features1 = get_cluster_center_features(
        image_embed1, cluster_labels1, n_clusters
    )
    center_features2 = get_cluster_center_features(
        image_embed2, cluster_labels2, n_clusters
    )
    
    if match_method == 'hungarian':
        mapping = hungarian_match_centers(center_features1, center_features2)
    elif match_method == 'argmin':
        mapping = argmin_matching(center_features1, center_features2)
    else:
        raise ValueError(f"Unknown match_method: {match_method}")
    
    return mapping


def match_centers_three_images(image_embeds, eigenvectors, match_method='hungarian'):
    """
    Match cluster centers across three images (A2 -> A1 -> B1).
    
    Args:
        image_embeds (torch.Tensor): Embeddings for 3 images [A2, A1, B1]
        eigenvectors (torch.Tensor): Eigenvectors for 3 images
        match_method (str): Matching method
        
    Returns:
        tuple: (A2_to_A1_mapping, A1_to_B1_mapping)
    """
    a2_to_a1_mapping = match_cluster_centers(
        image_embeds[0], image_embeds[1], 
        eigenvectors[0], eigenvectors[1], 
        match_method=match_method
    )
    
    a1_to_b1_mapping = match_cluster_centers(
        image_embeds[1], image_embeds[2], 
        eigenvectors[1], eigenvectors[2], 
        match_method=match_method
    )
    
    return a2_to_a1_mapping, a1_to_b1_mapping


def match_centers_two_images(image_embed1, image_embed2, eigvec1, eigvec2, 
                            match_method='hungarian'):
    return match_cluster_centers(
        image_embed1, image_embed2, eigvec1, eigvec2, match_method=match_method
    )


# ===== Visualization Functions =====

def plot_cluster_masks(image, eigenvector, cluster_order, hw=16):
    """
    blend the image with the cluster masks
    # image is (c, h, w)
    # eigenvector is (h*w, n_eig)
    # cluster_order is (n_eig), the order of the clusters
    """
    cluster_images = []
    base_img = img_transform_inv(image).resize(
        (128, 128), resample=Image.Resampling.NEAREST
    )
    
    for cluster_idx in cluster_order:
        # Create cluster mask
        cluster_mask = eigenvector.argmax(-1) == cluster_idx
        mask_array = cluster_mask.cpu().numpy()[1:].reshape(hw, hw)
        mask_array = (mask_array * 255).astype(np.uint8)
        
        # Resize mask to match image
        mask_img = Image.fromarray(mask_array).resize(
            (128, 128), resample=Image.Resampling.NEAREST
        )
        
        # Apply mask to image
        mask_normalized = np.array(mask_img).astype(np.float32) / 255
        img_array = np.array(base_img).astype(np.float32) / 255
        
        # Create 3-channel mask and apply
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        mask_3ch[mask_3ch == 0] = 0.1  # Dim non-masked areas
        
        masked_img = img_array * mask_3ch
        masked_img = (masked_img * 255).astype(np.uint8)
        
        cluster_images.append(Image.fromarray(masked_img))
    
    return cluster_images


def create_image_grid_row(image, eigenvector, cluster_order, discrete_colors, 
                         hw=16, n_cols=10):

    cluster_images = plot_cluster_masks(image, eigenvector, cluster_order, hw)
    
    # Prepare base images
    base_img = img_transform_inv(image).resize(
        (128, 128), resample=Image.Resampling.NEAREST
    )
    
    ncut_visualization = discrete_colors[1:].reshape(hw, hw, 3)
    ncut_img = Image.fromarray(ncut_visualization).resize(
        (128, 128), resample=Image.Resampling.NEAREST
    )
    
    # Pad cluster images to fill grid
    num_missing = n_cols - len(cluster_images) % n_cols
    if num_missing != n_cols:
        empty_img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        cluster_images.extend([empty_img] * num_missing)
    
    # Create grid rows
    prepend_images = [base_img, ncut_img]
    n_rows = len(cluster_images) // n_cols
    grid_rows = []
    
    for row_idx in range(n_rows):
        start_idx = row_idx * n_cols
        end_idx = (row_idx + 1) * n_cols
        row_images = prepend_images + cluster_images[start_idx:end_idx]
        grid_rows.append(row_images)
    
    return grid_rows


def create_multi_image_grid(images, eigenvectors, cluster_orders, discrete_colors, 
                           hw=16, n_cols=10):
    all_grid_rows = []
    
    for image, eigvec, cluster_order, discrete_rgb in zip(
        images, eigenvectors, cluster_orders, discrete_colors
    ):
        grid_rows = create_image_grid_row(
            image, eigvec, cluster_order, discrete_rgb, hw, n_cols
        )
        all_grid_rows.append(grid_rows)
    
    # Interleave rows from different images
    interleaved_rows = []
    for row_idx in range(len(all_grid_rows[0])):
        for img_idx in range(len(all_grid_rows)):
            interleaved_rows.append(all_grid_rows[img_idx][row_idx])
    
    return interleaved_rows


def get_correspondence_plot(images, eigenvectors, cluster_orders, discrete_colors, 
                           hw=16, n_cols=10):
    n_clusters = eigenvectors.shape[-1]
    n_cols = min(n_cols, n_clusters)
    
    interleaved_rows = create_multi_image_grid(
        images, eigenvectors, cluster_orders, discrete_colors, hw, n_cols
    )
    
    n_rows = len(interleaved_rows)
    n_cols = len(interleaved_rows[0])
    
    # Flatten all images and create final grid
    all_images = sum(interleaved_rows, [])
    final_grid = image_grid(all_images, n_rows, n_cols)
    
    return final_grid