"""
Color Palette Analyzer Module

Extract dominant color palettes from images, visualise swatches,
suggest complementary / analogous palettes, and compare two images
side-by-side.  Pure PIL + numpy — no model dependencies.
"""

import colorsys
import logging
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Palette extraction (mini-batch k-means on pixel RGB)
# ---------------------------------------------------------------------------

def _downsample(img: Image.Image, max_pixels: int = 40_000) -> np.ndarray:
    """Resize to cap total pixels, then return (N, 3) float32 array."""
    w, h = img.size
    ratio = (max_pixels / (w * h)) ** 0.5
    if ratio < 1:
        img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)
    return np.asarray(img.convert("RGB")).reshape(-1, 3).astype(np.float32)


def extract_palette(img: Image.Image, n_colors: int = 6, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Return *n_colors* dominant RGB tuples via k-means."""
    pixels = _downsample(img)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(pixels), size=min(n_colors, len(pixels)), replace=False)
    centers = pixels[indices].copy()

    for _ in range(20):
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(n_colors)
        ])
        if np.allclose(centers, new_centers, atol=1):
            break
        centers = new_centers

    counts = np.bincount(labels, minlength=n_colors)
    order = counts.argsort()[::-1]
    return [tuple(int(c) for c in centers[i]) for i in order]


# ---------------------------------------------------------------------------
# Colour-theory helpers
# ---------------------------------------------------------------------------

def _rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h, s, l


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def complementary_palette(palette: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    out = []
    for r, g, b in palette:
        h, s, l = _rgb_to_hsl(r, g, b)
        out.append(_hsl_to_rgb((h + 0.5) % 1.0, s, l))
    return out


def analogous_palette(palette: List[Tuple[int, int, int]], offset: float = 0.083) -> List[Tuple[int, int, int]]:
    out = []
    for r, g, b in palette:
        h, s, l = _rgb_to_hsl(r, g, b)
        out.append(_hsl_to_rgb((h + offset) % 1.0, s, l))
    return out


def triadic_palette(palette: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    out = []
    for r, g, b in palette:
        h, s, l = _rgb_to_hsl(r, g, b)
        out.append(_hsl_to_rgb((h + 1 / 3) % 1.0, s, l))
    return out


# ---------------------------------------------------------------------------
# Swatch rendering
# ---------------------------------------------------------------------------

_SWATCH_W = 80
_SWATCH_H = 80
_LABEL_H = 22
_GAP = 6


def _hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _render_palette_row(
    palette: List[Tuple[int, int, int]],
    label: str = "",
) -> Image.Image:
    """Draw a single row of colour swatches with hex labels."""
    n = len(palette)
    width = n * _SWATCH_W + (n - 1) * _GAP
    height = _SWATCH_H + _LABEL_H + (24 if label else 0)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y_off = 0
    if label:
        draw.text((0, 0), label, fill=(60, 60, 60))
        y_off = 24

    for i, rgb in enumerate(palette):
        x = i * (_SWATCH_W + _GAP)
        draw.rounded_rectangle(
            [x, y_off, x + _SWATCH_W, y_off + _SWATCH_H],
            radius=8,
            fill=rgb,
        )
        draw.text((x + 4, y_off + _SWATCH_H + 3), _hex(rgb), fill=(80, 80, 80))

    return canvas


def render_palette_image(
    palette: List[Tuple[int, int, int]],
    show_variations: bool = True,
) -> Image.Image:
    rows: List[Image.Image] = [_render_palette_row(palette, "Dominant")]
    if show_variations:
        rows.append(_render_palette_row(complementary_palette(palette), "Complementary"))
        rows.append(_render_palette_row(analogous_palette(palette), "Analogous"))
        rows.append(_render_palette_row(triadic_palette(palette), "Triadic"))

    max_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + _GAP * (len(rows) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + _GAP
    return canvas


# ---------------------------------------------------------------------------
# Comparison image (two palettes side-by-side)
# ---------------------------------------------------------------------------

def render_comparison(
    palette_a: List[Tuple[int, int, int]],
    palette_b: List[Tuple[int, int, int]],
) -> Image.Image:
    row_a = _render_palette_row(palette_a, "Image A — Dominant")
    row_b = _render_palette_row(palette_b, "Image B — Dominant")
    max_w = max(row_a.width, row_b.width)
    canvas = Image.new("RGB", (max_w, row_a.height + row_b.height + _GAP), (255, 255, 255))
    canvas.paste(row_a, (0, 0))
    canvas.paste(row_b, (0, row_a.height + _GAP))
    return canvas


# ---------------------------------------------------------------------------
# Gradio tab
# ---------------------------------------------------------------------------

def create_color_palette_tab():
    """Colour Palette Analyzer — standalone Gradio tab."""

    with gr.Tab("Color Palette"):
        gr.Markdown("## Color Palette Analyzer")
        gr.Markdown("Extract dominant colours, view complementary / analogous / triadic variations, and compare two images.")

        with gr.Row():
            img_a = gr.Image(label="Image A", type="pil")
            img_b = gr.Image(label="Image B (optional, for comparison)", type="pil")

        n_colors = gr.Slider(minimum=3, maximum=10, step=1, value=6, label="Number of colours to extract")

        analyze_btn = gr.Button("Analyze Palette", variant="primary")

        with gr.Row():
            palette_a_out = gr.Image(label="Image A Palette", interactive=False)
            palette_b_out = gr.Image(label="Image B Palette", interactive=False)

        comparison_out = gr.Image(label="Side-by-side comparison", interactive=False)

        # ── callback ────────────────────────────────────────────────────
        def _analyze(img_a_pil, img_b_pil, n):
            n = int(n)
            if img_a_pil is None:
                gr.Warning("Upload at least Image A.")
                return None, None, None

            pal_a = extract_palette(img_a_pil, n)
            out_a = render_palette_image(pal_a, show_variations=True)

            out_b = None
            comp = None
            if img_b_pil is not None:
                pal_b = extract_palette(img_b_pil, n)
                out_b = render_palette_image(pal_b, show_variations=True)
                comp = render_comparison(pal_a, pal_b)

            return out_a, out_b, comp

        analyze_btn.click(
            _analyze,
            inputs=[img_a, img_b, n_colors],
            outputs=[palette_a_out, palette_b_out, comparison_out],
        )
