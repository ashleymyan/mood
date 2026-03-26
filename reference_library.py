"""
Reference Library / Mood Board Module

A browsable gallery of reference-image pairs organised by mood / genre.
Users can filter by tag, bookmark pairs, and (future) load a pair into
the Poster Lab.  Bookmarks are stored in a local JSON file so they
persist across restarts without needing a remote backend.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Catalogue — each entry is a pair of images + metadata
# ---------------------------------------------------------------------------

_IMAGES_DIR = Path(__file__).parent / "images"
_BOOKMARKS_PATH = Path(__file__).parent / "bookmarks.json"

# Built-in catalogue derived from the repo's example images.
# Each entry: { id, title, tags[], image_a (path), image_b (path), description }
CATALOGUE: List[Dict] = [
    {
        "id": "violin-guitar",
        "title": "Musician Fusion",
        "tags": ["creative", "portrait", "warm"],
        "image_a": str(_IMAGES_DIR / "playviolin_hr.png"),
        "image_b": str(_IMAGES_DIR / "playguitar_hr.png"),
        "description": "Blend a violinist and guitarist vibe for a music-themed poster.",
    },
    {
        "id": "cat-bread",
        "title": "Whimsical Mash-up",
        "tags": ["whimsical", "fun", "surreal"],
        "image_a": str(_IMAGES_DIR / "input_cat.png"),
        "image_b": str(_IMAGES_DIR / "input_bread.png"),
        "description": "Playful surreal combination — cat meets bread.",
    },
    {
        "id": "02140",
        "title": "Abstract Pair 02140",
        "tags": ["abstract", "moody"],
        "image_a": str(_IMAGES_DIR / "02140_left.jpg"),
        "image_b": str(_IMAGES_DIR / "02140_right.jpg"),
        "description": "Abstract moody pairing with contrasting textures.",
    },
    {
        "id": "03969",
        "title": "Scene Pair 03969",
        "tags": ["cinematic", "warm"],
        "image_a": str(_IMAGES_DIR / "03969_l.jpg"),
        "image_b": str(_IMAGES_DIR / "03969_r.jpg"),
        "description": "Cinematic scene pair with warm, golden tones.",
    },
    {
        "id": "04963",
        "title": "Scene Pair 04963",
        "tags": ["cinematic", "cool"],
        "image_a": str(_IMAGES_DIR / "04963_l.jpg"),
        "image_b": str(_IMAGES_DIR / "04963_r.jpg"),
        "description": "Cool-toned cinematic scene pair.",
    },
    {
        "id": "00436",
        "title": "Scene Pair 00436",
        "tags": ["cinematic", "moody"],
        "image_a": str(_IMAGES_DIR / "00436_l.jpg"),
        "image_b": str(_IMAGES_DIR / "00436_r.jpg"),
        "description": "Moody cinematic composition.",
    },
    {
        "id": "archi",
        "title": "Architecture Blend",
        "tags": ["architecture", "geometric", "cool"],
        "image_a": str(_IMAGES_DIR / "archi" / "input_A.jpg"),
        "image_b": str(_IMAGES_DIR / "archi" / "input_B.jpg"),
        "description": "Architectural references with geometric patterns and clean lines.",
    },
    {
        "id": "bears",
        "title": "Pink vs Black Bear",
        "tags": ["whimsical", "contrast", "fun"],
        "image_a": str(_IMAGES_DIR / "pink_bear1.jpg"),
        "image_b": str(_IMAGES_DIR / "black_bear2.jpg"),
        "description": "Contrasting bear styles — playful pink meets dark moody.",
    },
]

ALL_TAGS = sorted({tag for entry in CATALOGUE for tag in entry["tags"]})


# ---------------------------------------------------------------------------
# Bookmark persistence (local JSON)
# ---------------------------------------------------------------------------

def _load_bookmarks() -> List[str]:
    if _BOOKMARKS_PATH.exists():
        try:
            return json.loads(_BOOKMARKS_PATH.read_text())
        except Exception:
            pass
    return []


def _save_bookmarks(ids: List[str]) -> None:
    try:
        _BOOKMARKS_PATH.write_text(json.dumps(ids))
    except Exception as exc:
        logging.warning(f"Could not save bookmarks: {exc}")


def _toggle_bookmark(entry_id: str) -> List[str]:
    bm = _load_bookmarks()
    if entry_id in bm:
        bm.remove(entry_id)
    else:
        bm.append(entry_id)
    _save_bookmarks(bm)
    return bm


# ---------------------------------------------------------------------------
# HTML card renderer
# ---------------------------------------------------------------------------

def _img_to_data_uri(path: str, max_size: int = 260) -> str:
    """Return a base64 data URI for an image, resized for thumbnail use."""
    import base64
    from io import BytesIO

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _render_catalogue_html(
    entries: List[Dict],
    bookmarks: List[str],
) -> str:
    if not entries:
        return "<p style='text-align:center; color:#888;'>No pairs match this filter.</p>"

    cards: List[str] = []
    for e in entries:
        is_bm = e["id"] in bookmarks
        bm_icon = "★" if is_bm else "☆"
        tag_chips = " ".join(
            f'<span style="background:#fce9dc;color:#7c4a3a;padding:2px 8px;border-radius:999px;font-size:0.78rem;">{t}</span>'
            for t in e["tags"]
        )
        uri_a = _img_to_data_uri(e["image_a"])
        uri_b = _img_to_data_uri(e["image_b"])
        cards.append(f"""
        <div style="border:1px solid #e8ddd4; border-radius:14px; padding:14px; background:#fffdf9;
                    box-shadow:0 4px 14px rgba(48,40,34,0.08); display:flex; flex-direction:column; gap:8px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <b style="font-size:1.05rem;">{e['title']}</b>
            <span style="font-size:1.3rem; cursor:default;" title="{'Bookmarked' if is_bm else 'Not bookmarked'}">{bm_icon}</span>
          </div>
          <div style="display:flex; gap:8px; justify-content:center;">
            <img src="{uri_a}" style="max-height:140px; border-radius:8px; border:1px solid #ddd;" />
            <img src="{uri_b}" style="max-height:140px; border-radius:8px; border:1px solid #ddd;" />
          </div>
          <div>{tag_chips}</div>
          <p style="margin:0; font-size:0.88rem; color:#6b5a4f;">{e['description']}</p>
          <code style="font-size:0.75rem; color:#aaa;">id: {e['id']}</code>
        </div>
        """)

    grid = "\n".join(cards)
    return f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:16px;">
      {grid}
    </div>
    """


# ---------------------------------------------------------------------------
# Gradio tab
# ---------------------------------------------------------------------------

def create_reference_library_tab():
    """Reference Library / Mood Board — standalone Gradio tab."""

    with gr.Tab("Reference Library"):
        gr.Markdown("## Reference Library")
        gr.Markdown("Browse curated reference pairs by mood and genre. Bookmark your favourites and load them into the Poster Lab.")

        with gr.Row():
            tag_filter = gr.Dropdown(
                choices=["All"] + ALL_TAGS,
                value="All",
                label="Filter by tag",
                scale=2,
            )
            bookmarks_only = gr.Checkbox(label="Bookmarked only", value=False)
            refresh_btn = gr.Button("Refresh", variant="secondary", elem_classes=["compact-btn"])

        gallery_html = gr.HTML(value="<p>Click <b>Refresh</b> to load the library.</p>")

        gr.Markdown("---")
        gr.Markdown("### Bookmark / Load a Pair")
        with gr.Row():
            pair_id_input = gr.Textbox(label="Pair ID", placeholder="e.g. violin-guitar")
            bookmark_btn = gr.Button("Toggle Bookmark", variant="secondary", elem_classes=["compact-btn"])
            load_btn = gr.Button("Preview Pair", variant="primary", elem_classes=["compact-btn"])

        bookmark_status = gr.Markdown("")

        with gr.Row():
            preview_a = gr.Image(label="Image A", interactive=False)
            preview_b = gr.Image(label="Image B", interactive=False)

        # ── callbacks ───────────────────────────────────────────────────
        def _refresh(tag, bm_only):
            bookmarks = _load_bookmarks()
            entries = CATALOGUE
            if tag and tag != "All":
                entries = [e for e in entries if tag in e["tags"]]
            if bm_only:
                entries = [e for e in entries if e["id"] in bookmarks]
            return _render_catalogue_html(entries, bookmarks)

        refresh_btn.click(
            _refresh,
            inputs=[tag_filter, bookmarks_only],
            outputs=[gallery_html],
        )
        tag_filter.change(
            _refresh,
            inputs=[tag_filter, bookmarks_only],
            outputs=[gallery_html],
        )
        bookmarks_only.change(
            _refresh,
            inputs=[tag_filter, bookmarks_only],
            outputs=[gallery_html],
        )

        def _toggle_bm(pair_id, tag, bm_only):
            pair_id = (pair_id or "").strip()
            if not pair_id:
                gr.Warning("Enter a pair ID first.")
                return "", _refresh(tag, bm_only)
            bm = _toggle_bookmark(pair_id)
            is_now = pair_id in bm
            status = f"**{pair_id}** {'bookmarked ★' if is_now else 'removed ☆'}"
            return status, _refresh(tag, bm_only)

        bookmark_btn.click(
            _toggle_bm,
            inputs=[pair_id_input, tag_filter, bookmarks_only],
            outputs=[bookmark_status, gallery_html],
        )

        def _load_pair(pair_id):
            pair_id = (pair_id or "").strip()
            match = next((e for e in CATALOGUE if e["id"] == pair_id), None)
            if match is None:
                gr.Warning(f"No pair with ID '{pair_id}' found.")
                return None, None
            try:
                a = Image.open(match["image_a"]).convert("RGB")
            except Exception:
                a = None
            try:
                b = Image.open(match["image_b"]).convert("RGB")
            except Exception:
                b = None
            return a, b

        load_btn.click(
            _load_pair,
            inputs=[pair_id_input],
            outputs=[preview_a, preview_b],
        )
