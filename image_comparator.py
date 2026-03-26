"""
Side-by-Side Image Comparator Module

Upload two images and get:
  - A slider-style split overlay
  - A pixel-difference heatmap
  - Dominant colour palettes for each
  - (Optional) DINO / CLIP cosine-similarity scores

Pure PIL + numpy for visuals.  Feature-similarity scores use the
existing extract_features helpers but are wrapped in a try/except so
the tab still works on CPU-only machines.
"""

import logging
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from color_palette_analyzer import extract_palette, render_comparison

# ---------------------------------------------------------------------------
# Visual comparison helpers
# ---------------------------------------------------------------------------

def _match_sizes(a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Resize both images to the same dimensions (max of each axis)."""
    w = max(a.width, b.width)
    h = max(a.height, b.height)
    a = a.resize((w, h), Image.Resampling.LANCZOS)
    b = b.resize((w, h), Image.Resampling.LANCZOS)
    return a, b


def create_split_view(a: Image.Image, b: Image.Image, position: float = 0.5) -> Image.Image:
    """Create a left/right split composite at *position* (0-1)."""
    a, b = _match_sizes(a, b)
    w, h = a.size
    split_x = int(w * position)

    canvas = Image.new("RGB", (w, h))
    left = a.crop((0, 0, split_x, h))
    right = b.crop((split_x, 0, w, h))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (split_x, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line([(split_x, 0), (split_x, h)], fill=(255, 255, 255), width=3)

    label_y = h - 30
    draw.text((10, label_y), "A", fill=(255, 255, 255))
    draw.text((w - 20, label_y), "B", fill=(255, 255, 255))

    return canvas


def create_diff_heatmap(a: Image.Image, b: Image.Image) -> Image.Image:
    """Absolute pixel difference mapped to a hot colourmap."""
    a, b = _match_sizes(a, b)
    arr_a = np.asarray(a.convert("RGB")).astype(np.float32)
    arr_b = np.asarray(b.convert("RGB")).astype(np.float32)

    diff = np.sqrt(((arr_a - arr_b) ** 2).sum(axis=2))
    if diff.max() > 0:
        diff = diff / diff.max()
    diff_uint8 = (diff * 255).astype(np.uint8)

    heatmap = Image.fromarray(diff_uint8, mode="L").convert("RGB")
    # tint toward warm colours
    r, g, b_ch = heatmap.split()
    r = r.point(lambda p: min(255, int(p * 1.4)))
    g = g.point(lambda p: int(p * 0.5))
    b_ch = b_ch.point(lambda p: int(p * 0.3))
    heatmap = Image.merge("RGB", (r, g, b_ch))
    return heatmap


def create_blend_overlay(a: Image.Image, b: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Alpha blend of the two images."""
    a, b = _match_sizes(a, b)
    return Image.blend(a.convert("RGBA"), b.convert("RGBA"), alpha).convert("RGB")


# ---------------------------------------------------------------------------
# Optional: DINO / CLIP similarity
# ---------------------------------------------------------------------------

def compute_similarity_scores(a: Image.Image, b: Image.Image) -> str:
    """Return a markdown string with cosine similarity from DINO and CLIP.

    Falls back gracefully when GPU / models are unavailable.
    """
    lines = []
    try:
        import torch
        from extract_features import (
            dino_image_transform,
            clip_image_transform,
            extract_dino_features,
            extract_clip_features,
        )

        def _cos(t1, t2):
            t1 = t1.flatten().float()
            t2 = t2.flatten().float()
            return torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

        imgs_dino = torch.stack([dino_image_transform(a), dino_image_transform(b)])
        dino_feats = extract_dino_features(imgs_dino)
        lines.append(f"**DINO cosine similarity:** {_cos(dino_feats[0], dino_feats[1]):.4f}")

        imgs_clip = torch.stack([clip_image_transform(a), clip_image_transform(b)])
        clip_feats = extract_clip_features(imgs_clip)
        lines.append(f"**CLIP cosine similarity:** {_cos(clip_feats[0], clip_feats[1]):.4f}")
    except Exception as exc:
        lines.append(f"_Feature similarity unavailable ({exc})_")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio tab
# ---------------------------------------------------------------------------

def create_image_comparator_tab():
    """Side-by-Side Comparator — standalone Gradio tab."""

    with gr.Tab("Image Comparator"):
        gr.Markdown("## Side-by-Side Comparator")
        gr.Markdown("Upload two images to compare them visually and analytically.")

        with gr.Row():
            img_a = gr.Image(label="Image A", type="pil")
            img_b = gr.Image(label="Image B", type="pil")

        with gr.Row():
            split_slider = gr.Slider(
                minimum=0.05, maximum=0.95, step=0.01, value=0.5,
                label="Split position (drag to reveal more of A or B)",
            )
            compute_features = gr.Checkbox(
                label="Compute DINO / CLIP similarity (needs GPU, slower)",
                value=False,
            )

        compare_btn = gr.Button("Compare", variant="primary")

        with gr.Row():
            split_out = gr.Image(label="Split View", interactive=False)
            diff_out = gr.Image(label="Difference Heatmap", interactive=False)

        blend_out = gr.Image(label="Alpha Blend Overlay (50 %)", interactive=False)
        palette_out = gr.Image(label="Palette Comparison", interactive=False)
        similarity_md = gr.Markdown("")

        # ── callback ────────────────────────────────────────────────────
        def _compare(a_pil, b_pil, split_pos, do_features):
            if a_pil is None or b_pil is None:
                gr.Warning("Please upload both images.")
                return None, None, None, None, ""

            split_img = create_split_view(a_pil, b_pil, split_pos)
            diff_img = create_diff_heatmap(a_pil, b_pil)
            blend_img = create_blend_overlay(a_pil, b_pil)

            pal_a = extract_palette(a_pil, 6)
            pal_b = extract_palette(b_pil, 6)
            palette_img = render_comparison(pal_a, pal_b)

            sim_text = ""
            if do_features:
                sim_text = compute_similarity_scores(a_pil, b_pil)

            return split_img, diff_img, blend_img, palette_img, sim_text

        compare_btn.click(
            _compare,
            inputs=[img_a, img_b, split_slider, compute_features],
            outputs=[split_out, diff_out, blend_out, palette_out, similarity_md],
        )

        split_slider.release(
            lambda a, b, pos: create_split_view(a, b, pos) if a is not None and b is not None else None,
            inputs=[img_a, img_b, split_slider],
            outputs=[split_out],
        )
