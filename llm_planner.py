from typing import Dict, Any, List
from io import BytesIO
import base64
import json
import os
import requests

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


def interpret_feedback_for_refinement(
    feedback_text: str,
    current_poster: Image.Image,
    current_style_notes: str = "",
    current_alpha_start: float = 0.0,
    current_alpha_end: float = 1.0,
    current_n_steps: int = 12,
    current_creative_prompt: str = "",
) -> Dict[str, Any]:
    """
    Interpret natural-language feedback on a generated poster and return adjusted
    Step-2 controls and poster style notes for the next iteration.

    Only the four Step-2 UI parameters (alpha_start, alpha_end, n_steps,
    creative_prompt) and the poster style_notes are ever modified — no internal
    vibe-space model parameters are touched.
    """
    poster_desc = _describe_image_with_vision(current_poster)

    system_prompt = (
        "You are an expert movie-poster art director helping a filmmaker iterate on a poster. "
        "There are two separate sets of parameters you can update:\n\n"
        "A) POSTER STYLE NOTES (for the AI image generator, affects look immediately):\n"
        "   Rewrite to directly address visual feedback "
        "   (e.g. 'too dark' -> 'brighter, high-key lighting'; "
        "   'more energy' -> 'dynamic composition, bold contrast').\n\n"
        "B) STEP-2 DRAFT CONTROLS (used when regenerating blended drafts):\n"
        "   - alpha_start / alpha_end: how far to interpolate between the two reference images. "
        "     Only change if the feedback is about the fundamental mood/vibe balance.\n"
        "   - n_steps: number of draft variants. Only change if user asks for more/fewer options.\n"
        "   - creative_prompt: ALWAYS update this to incorporate the user's feedback text "
        "     so the next draft generation reflects what the user wants.\n\n"
        "Constraints:\n"
        "- Return ONLY a single JSON object, no prose, no code fences.\n"
        "- JSON keys: updated_style_notes (string, under 200 chars), "
        "  updated_alpha_start (float), updated_alpha_end (float), "
        "  updated_n_steps (int), updated_creative_prompt (string), "
        "  explanation (string), suggest_redraft (bool).\n"
        "- alpha values in [-2.0, 2.0]; n_steps in [3, 40].\n"
        "- explanation: 1-2 sentences."
    )

    user_prompt = (
        f"Current poster description: {poster_desc}\n\n"
        f"User feedback: \"{feedback_text}\"\n\n"
        f"Current style notes: \"{current_style_notes or 'none'}\"\n"
        f"Current alpha_start: {current_alpha_start}, alpha_end: {current_alpha_end}\n"
        f"Current n_steps: {current_n_steps}\n"
        f"Current creative_prompt: \"{current_creative_prompt or 'none'}\"\n\n"
        "Update the poster style notes and, if warranted, the Step-2 draft controls."
    )

    try:
        resp = client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )
        raw = (resp.output_text or "").strip()
        content = json.loads(raw)
    except Exception:
        content = {
            "updated_style_notes": f"{current_style_notes}; {feedback_text}".strip("; "),
            "updated_alpha_start": current_alpha_start,
            "updated_alpha_end": current_alpha_end,
            "updated_n_steps": current_n_steps,
            "updated_creative_prompt": current_creative_prompt,
            "explanation": "Applied feedback directly to style notes.",
            "suggest_redraft": False,
        }

    updated_alpha_start = max(-2.0, min(2.0, float(content.get("updated_alpha_start", current_alpha_start))))
    updated_alpha_end = max(-2.0, min(2.0, float(content.get("updated_alpha_end", current_alpha_end))))
    raw_steps = content.get("updated_n_steps", current_n_steps)
    updated_n_steps = max(3, min(40, int(raw_steps) if raw_steps is not None else current_n_steps))

    return {
        "updated_style_notes": str(content.get("updated_style_notes", current_style_notes or "")),
        "updated_alpha_start": updated_alpha_start,
        "updated_alpha_end": updated_alpha_end,
        "updated_n_steps": updated_n_steps,
        "updated_creative_prompt": str(content.get("updated_creative_prompt", current_creative_prompt or "")),
        "explanation": str(content.get("explanation", "")),
        "suggest_redraft": bool(content.get("suggest_redraft", False)),
    }


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
      2. Ask GPT-4.1 to craft a detailed Ideogram prompt that weaves in the text.
      3. Call Ideogram API and return the result as a PIL Image.
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

    # ── 3. Ask GPT to craft the Ideogram prompt ──────────────────────────
    system_prompt = (
        "You are a professional movie-poster art director. "
        "Given a visual style description and text elements, craft a vivid, specific Ideogram prompt "
        "for a cinematic movie poster. "
        "The poster MUST visually show every piece of text exactly as given—rendered in a legible, "
        "stylish font appropriate to the mood. "
        "Describe composition, lighting, color palette, typography placement, and atmosphere. "
        "Ideogram excels at typography, so ensure text is prominent and readable. "
        "Return ONLY the Ideogram prompt; no explanation, no preamble."
    )
    user_prompt = (
        f"Reference image style: {style_desc}\n\n"
        f"Text elements to embed in the poster: {text_summary}\n\n"
        f"Additional style notes from user: {style_notes or 'none'}\n\n"
        "Create an Ideogram prompt for a professional movie poster that matches this visual style "
        "and prominently features all text elements, legibly rendered."
    )

    plan_resp = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    ideogram_prompt = (plan_resp.output_text or "").strip()

    if not ideogram_prompt:
        ideogram_prompt = f"A cinematic movie poster featuring: {text_summary}. Style: {style_desc}"

    # Prepend a safety prefix so Ideogram knows this is art
    ideogram_prompt = "Cinematic movie poster art. " + ideogram_prompt

    # ── 4. Generate with Ideogram API ────────────────────────────────
    api_key = os.environ.get("IDEOGRAM_API_KEY")
    if not api_key:
        raise ValueError("IDEOGRAM_API_KEY environment variable not set")

    response = requests.post(
        "https://api.ideogram.ai/v1/ideogram-v3/generate",
        headers={"Api-Key": api_key},
        files={
            "prompt": (None, ideogram_prompt),
            "rendering_speed": (None, "DEFAULT"),
        }
    )
    response.raise_for_status()
    data = response.json()
    image_url = data["data"][0]["url"]

    # Download the image
    img_response = requests.get(image_url)
    img_response.raise_for_status()
    img_bytes = img_response.content
    return Image.open(BytesIO(img_bytes)).convert("RGB")
