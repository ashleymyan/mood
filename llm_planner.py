from typing import Dict, Any, List
from io import BytesIO
import base64
import json
import os

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

client = OpenAI()

def _encode_image_to_data_url(img: Image.Image, max_side: int = 768) -> str:
    """
    Convert a PIL Image to a base64 data URL.
    Also downsizes to keep requests small and reliable.
    """
    img = img.convert("RGB")
    img = img.copy()
    img.thumbnail((max_side, max_side))

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _describe_image_with_vision(img: Image.Image) -> str:
    """
    Use a vision-capable model to get a short, semantic description of the image.
    If anything fails, fall back to a very generic description.
    """

    try:
        data_url = _encode_image_to_data_url(img)

        resp = client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text",
                     "text": "Describe this image in 1–2 short sentences, focusing on subject, pose, colors, style, and overall vibe."},
                    {"type": "input_image",
                     "image_url": data_url},
                ],
            }],
        )
    
        desc = resp.output_text
        if not desc:
            raise ValueError("Empty caption from model.")
        return desc.strip()

    except Exception:
        # Safe fallback so the rest of the pipeline still works
        w, h = img.size
        mode = img.mode
        return (
            f"An image of size {w}x{h} with mode {mode}. "
            f"The detailed visual content could not be automatically described."
        )


def analyze_pair_with_llm(
    img1: Image.Image,
    img2: Image.Image,
    creative_prompt: str = "",
) -> Dict[str, Any]:
    """
    Use GPT to:
      - reason about what might be shared between the two images
      - reason about what might be shared between the two images (via captions)
      - infer interesting attributes to blend
      - suggest interpolation parameters for creativity
    Returns a dict with fields used by app.py:
      - focus_attributes: List[str]
      - alpha_start: float
      - alpha_end: float
      - n_steps: int | None
      - explanation: str
    """

    img1_desc = _describe_image_with_vision(img1)
    img2_desc = _describe_image_with_vision(img2)

     # Default creative prompt
    if not creative_prompt:
        creative_prompt = (
            "Be moderately creative while keeping both identities recognizable."
        )

    system_prompt = (
        "You are a visual creativity planner for a vibe-blending image model. "
        "Given short descriptions of two images and a user creativity intent, "
        "you must decide:\n"
        "1) Which 1–3 shared or related visual attributes would be most interesting "
        "   to emphasize when blending (e.g., hair shape, pose, silhouette, color mood).\n"
        "2) How strong the interpolation / extrapolation should be: "
        "   alpha_start (usually 0.0) and alpha_end (0.8–1.5), where >1 means extrapolation.\n"
        "3) Whether to increase the number of steps (n_steps) for smoother, more creative transitions.\n\n"
        "Constraints:\n"
        "- Return ONLY a single JSON object, no prose before or after it.\n"
        "- Do NOT wrap the JSON in code fences.\n"
        "- JSON keys: focus_attributes (list of short strings), "
        "  alpha_start (float), alpha_end (float), n_steps (int or null), explanation (string).\n"
        "- alpha_start should usually be 0.0 or slightly negative if the user wants extreme creativity.\n"
        "- alpha_end should be between 0.4 and 1.5.\n"
        "- n_steps should be between 6 and 40 if you choose to override; "
        "set it to null if you want to keep the UI default.\n"
        "- explanation should be 1–3 sentences, plain text."
    )

    user_prompt = f"""
    Here are the two images (described):

    Image 1: {img1_desc}
    Image 2: {img2_desc}

    User's creative intent: "{creative_prompt}"

    Based on this, choose:
    - which key visual attributes or vibes to emphasize when blending,
    - how far to interpolate/extrapolate (alpha_start, alpha_end),
    - whether to override the number of steps (n_steps) for a creative but still coherent sequence.
    """

    resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )

    raw_content = (resp.output_text or "").strip()

    try:
        content = json.loads(raw_content)
    except Exception:
        content = {
            "focus_attributes": [],
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "n_steps": None,
            "explanation": (
                "Using default interpolation settings because the planner failed "
                "to return a valid JSON plan."
            ),
        }

    # Final safety clamps
    alpha_start = float(content.get("alpha_start", 0.0))
    alpha_end = float(content.get("alpha_end", 1.0))
    n_steps = content.get("n_steps", None)

    alpha_start = max(-2.0, min(2.0, alpha_start))
    alpha_end = max(-2.0, min(2.0, alpha_end))

    if isinstance(n_steps, int):
        if n_steps < 3:
            n_steps = 3
        if n_steps > 40:
            n_steps = 40
    else:
        n_steps = None

    suggestion = {
        "focus_attributes": content.get("focus_attributes", []),
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "n_steps": n_steps,
        "explanation": content.get("explanation", ""),
    }

    return suggestion


def judge_best_blend(
    img1: Image.Image,
    img2: Image.Image,
    blended_images: list[Image.Image],
    creative_prompt: str,
    judge_criteria: str = ""
) -> dict:
    """
    Returns: {"best_index": int, "reason": str}
    Uses GPT-Vision to rank blended outputs.
    """
    if not blended_images:
        return {"best_index": 0, "reason": "No candidates to judge."}

    # Keep it cheap: judge at most first 12 candidates
    candidates = blended_images[:12]

    # Caption inputs once (optional)
    a_desc = _describe_image_with_vision(img1)
    b_desc = _describe_image_with_vision(img2)

    # Encode candidate images
    candidate_urls = [_encode_image_to_data_url(im) for im in candidates]

    if not judge_criteria:
        judge_criteria = "Most creative while still coherent and aligned with the creative intent."

    system = (
        "You are a strict evaluator for vibe-blending results. "
        "Pick the single best candidate according to the given criteria. "
        "Return ONLY JSON: {\"best_index\": <int>, \"reason\": <string>}."
    )

    # Build a single message containing all candidates
    content = [{"type": "input_text", "text": (
        f"Input A: {a_desc}\n"
        f"Input B: {b_desc}\n\n"
        f"Creative intent: {creative_prompt}\n"
        f"Judge criteria: {judge_criteria}\n\n"
        "Below are candidate blended images. Pick the best one.\n"
        f"Return best_index as an integer from 0 to {len(candidates)-1}."
    )}]

    for i, url in enumerate(candidate_urls):
        content.append({"type": "input_text", "text": f"Candidate {i}:"})
        content.append({"type": "input_image", "image_url": url})

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": content},
        ],
    )

    raw = (resp.output_text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        
    try:
        out = json.loads(raw)
        best = int(out.get("best_index", 0))
        best = max(0, min(len(candidates)-1, best))
        reason = str(out.get("reason", "")).strip()
        return {"best_index": best, "reason": reason}
    except Exception:
        return {"best_index": 0, "reason": "Judge failed to return valid JSON; defaulting to candidate 0."}


# ---------------------------------------------------------------------------
# Poster text helpers
# ---------------------------------------------------------------------------

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a truetype font at the given size, falling back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_text_overlay_poster(
    base_image: Image.Image,
    title: str = "",
    tagline: str = "",
    director: str = "",
    cast: str = "",
    release_year: str = "",
) -> Image.Image:
    """
    Composite text onto the base image to produce a movie-poster layout.
    Title + tagline appear in the lower third; director/cast/year at the very bottom.
    """
    img = base_image.copy().convert("RGBA")
    w, h = img.size

    # ── dark gradient bands at top (for credits row) and bottom (for title block) ──
    band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bottom_band_h = h // 3
    bd.rectangle([(0, h - bottom_band_h), (w, h)], fill=(0, 0, 0, 175))
    bd.rectangle([(0, 0), (w, h // 8)], fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, band).convert("RGB")

    draw = ImageDraw.Draw(img)

    def _shadow_text(x: int, y: int, text: str, font: ImageFont.FreeTypeFont,
                     fill=(255, 255, 255), shadow=(0, 0, 0), offset: int = 2):
        draw.text((x + offset, y + offset), text, font=font, fill=shadow)
        draw.text((x, y), text, font=font, fill=fill)

    def _center_text(y: int, text: str, font: ImageFont.FreeTypeFont,
                     fill=(255, 255, 255)):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = max(16, (w - text_w) // 2)
        _shadow_text(x, y, text, font, fill=fill)
        return bbox[3] - bbox[1]  # return line height

    y_cursor = h - bottom_band_h + 18

    # Title
    if title:
        title_size = max(48, w // 10)
        title_font = _get_font(title_size)
        line_h = _center_text(y_cursor, title.upper(), title_font)
        y_cursor += line_h + 10

    # Tagline
    if tagline:
        tag_size = max(20, w // 28)
        tag_font = _get_font(tag_size)
        line_h = _center_text(y_cursor, tagline, tag_font, fill=(220, 210, 200))
        y_cursor += line_h + 14

    # Credits block at very bottom
    credit_lines = []
    if director:
        credit_lines.append(f"Directed by {director}")
    if cast:
        credit_lines.append(cast)
    if release_year:
        credit_lines.append(str(release_year))

    if credit_lines:
        cred_size = max(14, w // 55)
        cred_font = _get_font(cred_size)
        cy = h - 12
        for line in reversed(credit_lines):
            bbox = draw.textbbox((0, 0), line, font=cred_font)
            lh = bbox[3] - bbox[1]
            cy -= lh + 6
            _center_text(cy, line, cred_font, fill=(180, 170, 160))

    return img


def generate_poster_with_text(
    base_image: Image.Image,
    title: str = "",
    tagline: str = "",
    director: str = "",
    cast: str = "",
    release_year: str = "",
    style_notes: str = "",
) -> Image.Image:
    """
    Generate a new AI-created movie poster that incorporates the provided text
    and matches the visual style of *base_image*.

    Steps:
      1. Describe the base image's visual style with GPT-4o.
      2. Ask GPT-4.1 to craft a detailed DALL-E 3 prompt that weaves in the text.
      3. Call DALL-E 3 and return the result as a PIL Image.
    """
    # ── 1. Describe the style of the reference image ──────────────────────
    style_desc = _describe_image_with_vision(base_image)

    # ── 2. Build the list of text elements ────────────────────────────────
    text_parts: List[str] = []
    if title:
        text_parts.append(f'film title "{title}"')
    if tagline:
        text_parts.append(f'tagline "{tagline}"')
    if director:
        text_parts.append(f'director credit "Directed by {director}"')
    if cast:
        text_parts.append(f'cast listing "{cast}"')
    if release_year:
        text_parts.append(f'release year "{release_year}"')
    text_summary = "; ".join(text_parts) if text_parts else "no specific text required"

    # ── 3. Ask GPT to craft the DALL-E 3 prompt ──────────────────────────
    system_prompt = (
        "You are a professional movie-poster art director. "
        "Given a visual style description and text elements, craft a vivid, specific DALL-E 3 prompt "
        "for a cinematic movie poster. "
        "The poster MUST visually show every piece of text exactly as given—rendered in a legible, "
        "stylish font appropriate to the mood. "
        "Describe composition, lighting, color palette, typography placement, and atmosphere. "
        "Return ONLY the DALL-E prompt; no explanation, no preamble."
    )
    user_prompt = (
        f"Reference image style: {style_desc}\n\n"
        f"Text elements to embed in the poster: {text_summary}\n\n"
        f"Additional style notes from user: {style_notes or 'none'}\n\n"
        "Create a DALL-E 3 prompt for a professional movie poster that matches this visual style "
        "and prominently features all text elements, legibly rendered."
    )

    plan_resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    dalle_prompt = (plan_resp.output_text or "").strip()

    # Prepend a safety prefix so DALL-E knows this is art
    dalle_prompt = "Cinematic movie poster art. " + dalle_prompt

    # ── 4. Generate with DALL-E 3 (b64 to avoid download request) ────────
    img_resp = client.images.generate(
        model="dall-e-3",
        prompt=dalle_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        response_format="b64_json",
    )
    b64_data = img_resp.data[0].b64_json
    img_bytes = base64.b64decode(b64_data)
    return Image.open(BytesIO(img_bytes)).convert("RGB")
