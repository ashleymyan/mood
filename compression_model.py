"""
Neural Compression Model for Feature Space Learning

This module implements a compression model that learns to compress and decompress
image features while preserving their geometric and semantic properties using
normalized cuts (NCut) and various geometric losses.
"""

import gc
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from omegaconf import DictConfig
import gradio as gr

# NCut and geometric utilities
from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
from ncut_pytorch import ncut_fn, kway_ncut
from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut
from ncut_pytorch.utils.math import rbf_affinity

# # Geometric loss functions
# from riemann_curvature_loss import (
#     compute_riemann_curvature_loss, 
#     compute_boundary_loss, 
#     compute_repulsion_loss,
#     compute_axis_align_loss
# )


# ===== Loss Functions =====

def compute_kway_ncut_loss(ground_truth_eigvec, predicted_eigvec, n_eig: int) -> torch.Tensor:
    """
    Compute k-way normalized cut loss between ground truth and predicted eigenvectors.
    
    Args:
        ground_truth_eigvec (torch.Tensor): Ground truth eigenvectors
        predicted_eigvec (torch.Tensor): Predicted eigenvectors
        n_eig (int): Number of eigenvectors to use
        
    Returns:
        torch.Tensor: Smooth L1 loss between Gram matrices
    """
    gt_subset = ground_truth_eigvec[:, :n_eig]
    pred_subset = predicted_eigvec[:, :n_eig]
    
    # Compute Gram matrices and compare them
    gt_gram = gt_subset @ gt_subset.T
    pred_gram = pred_subset @ pred_subset.T
    
    return F.smooth_l1_loss(gt_gram, pred_gram)


def compute_eigenvector_loss(ground_truth_eigvec, predicted_eigvec, n_eig: int, 
                           start: int = 4, step_mult: int = 2) -> torch.Tensor:
    """
    Compute eigenvector loss by aggregating k-way NCut losses across multiple scales.
    
    Args:
        ground_truth_eigvec (torch.Tensor): Ground truth eigenvectors
        predicted_eigvec (torch.Tensor): Predicted eigenvectors
        n_eig (int): Maximum number of eigenvectors
        start (int): Starting number of eigenvectors
        step_mult (int): Multiplication factor for scaling
        
    Returns:
        torch.Tensor: Aggregated eigenvector loss
    """
    # Handle edge cases
    if torch.all(ground_truth_eigvec == 0) or torch.all(predicted_eigvec == 0):
        return torch.tensor(0.0, device=ground_truth_eigvec.device)
    
    total_loss = 0.0
    current_n_eig = start // step_mult
    
    while True:
        current_n_eig *= step_mult
        total_loss += compute_kway_ncut_loss(
            ground_truth_eigvec, predicted_eigvec, current_n_eig
        )
        
        # Stop if we exceed the available eigenvectors
        max_available = min(ground_truth_eigvec.shape[1], predicted_eigvec.shape[1])
        if current_n_eig > max_available:
            break
    
    return total_loss


def compute_ncut_eigenvectors(features: torch.Tensor, n_eig: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function to compute NCut eigenvectors using RBF affinity.
    
    Args:
        features (torch.Tensor): Input features
        n_eig (int): Number of eigenvectors to compute
        gamma (float): RBF kernel parameter
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Eigenvectors and eigenvalues
    """
    gamma = features.var(0).sum().item()
    affinity_matrix = rbf_affinity(features, gamma=gamma)
    eigenvectors, eigenvalues = _plain_ncut(affinity_matrix, n_eig)
    return eigenvectors, eigenvalues


# ===== Neural Network Components =====

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron with GELU activations.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        num_layers (int): Number of hidden layers
        hidden_dim (int): Hidden layer dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, hidden_dim: int = 4096):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SpatialPoolingCNN(nn.Module):
    """
    CNN layer for spatial pooling of feature maps with support for sequence inputs.
    
    Handles inputs with CLS tokens and reshapes appropriately for 2D convolution.
    
    Args:
        num_channels (int): Number of input/output channels
        downsample_factor (int): Downsampling factor for pooling
    """
    
    def __init__(self, num_channels: int, downsample_factor: int = 4):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.conv = nn.Conv2d(
            num_channels, num_channels, 
            kernel_size=downsample_factor, 
            stride=downsample_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both (batch, seq_len, channels) and (seq_len, channels) inputs.
        """
        # Handle input shape variations
        added_batch_dim = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch_dim = True
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (B, L, C) or (L, C), got {x.shape}")

        batch_size, seq_len, channels = x.shape
        
        if seq_len < 2:
            raise ValueError("Sequence length must be at least 2 (1 CLS token + 1 patch)")

        # Validate that seq_len-1 is a perfect square (for spatial arrangement)
        spatial_size = int(round((seq_len - 1) ** 0.5))
        if spatial_size * spatial_size != (seq_len - 1):
            raise ValueError(f"seq_len-1 must be perfect square. Got {seq_len-1}")

        # Separate CLS token and spatial features
        cls_tokens = x[:, :1, :]  # (B, 1, C)
        spatial_features = x[:, 1:, :]  # (B, H*W, C)
        
        # Reshape to 2D for convolution
        spatial_2d = rearrange(
            spatial_features, 'b (h w) c -> b c h w', 
            h=spatial_size, w=spatial_size
        )
        
        # Apply pooling
        pooled_features = self.conv(spatial_2d)
        
        # Reshape back to sequence format
        pooled_sequence = rearrange(pooled_features, 'b c h w -> b (h w) c')
        
        # Concatenate CLS token back
        output = torch.cat([cls_tokens, pooled_sequence], dim=1)

        # Remove batch dimension if it was added
        if added_batch_dim:
            output = output.squeeze(0)
        
        return output


class MLPWithSpatialPooling(nn.Module):
    """
    MLP with integrated spatial pooling for handling sequence data with spatial structure.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        num_layers (int): Number of hidden layers
        hidden_dim (int): Hidden layer dimension
        downsample_factor (int): Spatial downsampling factor
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, 
                 hidden_dim: int = 4096, downsample_factor: int = 4):
        super().__init__()
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            SpatialPoolingCNN(hidden_dim, downsample_factor),
            nn.GELU()
        ]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===== Main Compression Model =====

class CompressionModel(pl.LightningModule):
    """
    Neural compression model for learning compressed feature representations.
    
    This model compresses input features to a lower-dimensional "mood space" and
    then decompresses them back, while preserving geometric and semantic properties
    through various loss functions including NCut-based losses and geometric constraints.
    
    Args:
        config: Configuration object containing model hyperparameters
        enable_gradio_progress (bool): Whether to show progress in Gradio interface
        use_identity_mapping (bool): Whether to use identity mapping for reconstruction
    """
    
    def __init__(self, config: DictConfig, enable_gradio_progress: bool = False, 
                 use_identity_mapping: bool = True):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.use_identity_mapping = use_identity_mapping
        self.downsample_factor = 2
        
        # Build encoder-decoder architecture
        self.encoder = MultiLayerPerceptron(
            config.in_dim, config.mood_dim, config.n_layer, config.latent_dim
        )
        
        self.decoder = MLPWithSpatialPooling(
            config.mood_dim, config.out_dim, config.n_layer, 
            config.latent_dim, self.downsample_factor
        )
        
        # Optional identity mapping decoder
        if self.use_identity_mapping:
            self.identity_decoder = MultiLayerPerceptron(
                config.mood_dim, config.in_dim, config.n_layer, config.latent_dim
            )

        # Training utilities
        self.loss_history = defaultdict(list)
        self.enable_gradio_progress = enable_gradio_progress
        if enable_gradio_progress:
            self.progress_tracker = gr.Progress()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder."""
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def training_step(self, batch, batch_idx):
        """Single training step with multiple loss components."""
        
        # Update progress if using Gradio
        if (self.enable_gradio_progress and 
            self.trainer.global_step % 10 == 0 and 
            self.trainer.global_step > 0 and
            self.loss_history['recon']):
            
            progress = self.trainer.global_step / self.config.steps
            recent_loss = self.loss_history['recon'][-1]
            self.progress_tracker(progress, desc=f"Training, loss = {recent_loss:.4f}")

        # Unpack batch
        input_features, target_features = batch
        
        # Forward pass
        compressed_features = self.encoder(input_features)
        reconstructed_features = self.decoder(compressed_features)
        
        # Optional identity mapping
        if self.use_identity_mapping:
            identity_reconstructed = self.identity_decoder(compressed_features)
        
        # Compute NCut eigenvectors for geometric consistency
        gt_eigenvectors = self._compute_ncut_eigenvectors(input_features)
        pred_eigenvectors = self._compute_ncut_eigenvectors(compressed_features)
        
        # Aggregate all loss components
        total_loss = self._compute_total_loss(
            input_features, target_features, compressed_features,
            reconstructed_features, identity_reconstructed if self.use_identity_mapping else None,
            gt_eigenvectors, pred_eigenvectors
        )
        
        self.log("loss/total", total_loss, prog_bar=True)
        return total_loss
    
    def _compute_ncut_eigenvectors(self, features: torch.Tensor) -> torch.Tensor:
        """Compute NCut eigenvectors for features."""
        # Flatten batch and sequence dimensions
        flattened_features = features.flatten(0, 1)
        if len(flattened_features) > 0:
            eigenvectors, _ = compute_ncut_eigenvectors(flattened_features, self.config.n_eig)
            return eigenvectors
        else:
            # Return zero tensor if no features
            return torch.zeros((1, self.config.n_eig), device=features.device)
    
    def _compute_multiscale_similarity(self, eigenvectors: torch.Tensor, 
                                      start_n_eig: int = 2, step_mult: int = 2) -> torch.Tensor:
        """
        Compute multi-scale similarity matrix from eigenvectors.
        
        Aggregates normalized Gram matrices across multiple scales by doubling
        the number of eigenvectors at each step.
        
        Args:
            eigenvectors (torch.Tensor): Eigenvectors of shape (n_samples, n_eig)
            start_n_eig (int): Starting number of eigenvectors
            step_mult (int): Multiplication factor for scaling
            
        Returns:
            torch.Tensor: Averaged similarity matrix
        """
        total_similarity = 0.0
        num_scales = 0
        current_n_eig = start_n_eig
        max_available = eigenvectors.shape[1]
        
        while current_n_eig <= max_available:
            # Extract subset of eigenvectors and normalize
            eigvec_subset = eigenvectors[:, :current_n_eig]
            eigvec_normalized = F.normalize(eigvec_subset, dim=-1)
            
            # Compute Gram matrix (similarity)
            total_similarity += eigvec_normalized @ eigvec_normalized.T
            num_scales += 1
            
            # Scale up for next iteration
            current_n_eig *= step_mult
        
        # Average across scales
        return total_similarity / num_scales if num_scales > 0 else total_similarity
    
    def _compute_total_loss(self, input_features, target_features, compressed_features,
                           reconstructed_features, identity_reconstructed, 
                           gt_eigenvectors, pred_eigenvectors):
        """Compute and aggregate all loss components."""
        total_loss = 0.0

        # Flag space loss - preserves multi-scale spectral structure
        if self.config.flag_loss > 0:
            # Compute ground truth similarity from eigenvectors
            gt_similarity = self._compute_multiscale_similarity(gt_eigenvectors)
            
            # Compute predicted similarity from compressed features
            flattened_compressed = compressed_features.flatten(0, 1)
            pred_similarity = flattened_compressed @ flattened_compressed.T
            
            # Compare similarity matrices
            flag_loss = F.smooth_l1_loss(gt_similarity, pred_similarity)
            self.log("loss/flag", flag_loss, prog_bar=True)
            total_loss += flag_loss * self.config.flag_loss
            self.loss_history['flag'].append(flag_loss.item())
        
        # Eigenvector preservation loss
        if self.config.eigvec_loss > 0:
            eigvec_loss = compute_eigenvector_loss(
                gt_eigenvectors, pred_eigenvectors, n_eig=self.config.n_eig
            )
            self.log("loss/eigvec", eigvec_loss, prog_bar=True)
            total_loss += eigvec_loss * self.config.eigvec_loss
            self.loss_history['eigvec'].append(eigvec_loss.item())

        # Reconstruction loss
        if self.config.recon_loss > 0:
            recon_loss = F.smooth_l1_loss(target_features, reconstructed_features)
            self.log("loss/recon", recon_loss, prog_bar=True)
            total_loss += recon_loss * self.config.recon_loss
            self.loss_history['recon'].append(recon_loss.item())

        # Identity mapping loss
        if self.use_identity_mapping and self.config.recon_loss_id > 0:
            id_loss = F.smooth_l1_loss(input_features, identity_reconstructed)
            self.log("loss/recon_id", id_loss, prog_bar=True)
            total_loss += id_loss * self.config.recon_loss_id
            id_loss = F.smooth_l1_loss(
                input_features, 
                identity_reconstructed
            )
            self.log("loss/recon_id", id_loss, prog_bar=True)
            total_loss += id_loss * self.config.recon_loss_id

        # # Geometric losses on compressed features
        # if self.config.riemann_curvature_loss > 0:
        #     curvature_loss = compute_riemann_curvature_loss(compressed_features)
        #     self.log("loss/riemann_curvature", curvature_loss, prog_bar=True)
        #     total_loss += curvature_loss * self.config.riemann_curvature_loss

        # if self.config.axis_align_loss > 0:
        #     axis_loss = compute_axis_align_loss(compressed_features)
        #     self.log("loss/axis_align", axis_loss, prog_bar=True)
        #     total_loss += axis_loss * self.config.axis_align_loss

        # if self.config.repulsion_loss > 0:
        #     repulsion_loss = compute_repulsion_loss(compressed_features)
        #     self.log("loss/repulsion", repulsion_loss, prog_bar=True)
        #     total_loss += repulsion_loss * self.config.repulsion_loss

        # if self.config.boundary_loss > 0:
        #     flattened_compressed = rearrange(compressed_features, 'b l c -> (b l) c')
        #     boundary_loss = compute_boundary_loss(flattened_compressed)
        #     self.log("loss/boundary", boundary_loss, prog_bar=True)
        #     total_loss += boundary_loss * self.config.boundary_loss

        return total_loss
    
    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.NAdam(self.parameters(), lr=self.config.lr)


# ===== Dataset and Training Utilities =====

class FeatureDataset(torch.utils.data.Dataset):
    """
    Dataset for feature compression training.
    
    Args:
        input_features (torch.Tensor): Input feature tensors
        target_features (torch.Tensor): Target feature tensors
    """
    
    def __init__(self, input_features: torch.Tensor, target_features: torch.Tensor):
        self.input_features = input_features
        self.target_features = target_features
    
    def __len__(self) -> int:
        return len(self.input_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.input_features[idx], 
            self.target_features[idx]
        )


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def train_compression_model(model: CompressionModel, 
                          config: DictConfig,
                          input_features: torch.Tensor,
                          target_features: torch.Tensor, 
                          devices: List[int] = [0]) -> pl.Trainer:
    """
    Train the compression model with the given data.
    
    Args:
        model: The compression model to train
        config: Training configuration
        input_features: Input feature tensors
        target_features: Target feature tensors
        devices: GPU devices to use
        
    Returns:
        pl.Trainer: The trained PyTorch Lightning trainer
    """
    clear_gpu_memory()

    # Create dataset and dataloader
    dataset = FeatureDataset(input_features, target_features)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_steps=config.steps,
        gradient_clip_val=config.grad_clip_val,
        accelerator="gpu", 
        devices=devices,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False  # Disable default logger
    )
    
    # Train the model
    trainer.fit(model, dataloader)
    
    return trainer
