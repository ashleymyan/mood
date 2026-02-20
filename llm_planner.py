from typing import Dict, Any
from PIL import Image
from openai import OpenAI
import json

client = OpenAI()

def _summarize_image_placeholder(img: Image.Image) -> str:
    """
    For now, we don't send raw image pixels to GPT.
    This placeholder can be replaced with real vision model calls later.
    For now, we just include rough metadata.
    """
    w, h = img.size
    mode = img.mode
    return f"An image of size {w}x{h} with mode {mode}. (Visual content not directly described.)"


def analyze_pair_with_llm(
    img1: Image.Image,
    img2: Image.Image,
    creative_prompt: str = "",
) -> Dict[str, Any]:
    """
    Use GPT to:
      - reason about what might be shared between the two images
      - infer interesting attributes to blend
      - suggest interpolation parameters for creativity
    Returns a dict with fields used by app.py.
    """

    # Can later replace these with actual image captions (via another vision model)
    img1_desc = _summarize_image_placeholder(img1)
    img2_desc = _summarize_image_placeholder(img2)

    if not creative_prompt:
        creative_prompt = (
            "Be moderately creative while keeping both identities recognizable."
        )

    system_prompt = (
        "You are a visual creativity planner for a vibe-blending image model. "
        "Given rough descriptions of two images and a user creativity intent, "
        "you must decide:\n"
        "1) Which 1–3 shared or related visual attributes would be most interesting "
        "   to emphasize when blending (e.g., hair shape, pose, silhouette, color mood).\n"
        "2) How strong the interpolation / extrapolation should be: "
        "   alpha_start (usually 0.0) and alpha_end (0.8–1.5), where >1 means extrapolation.\n"
        "3) Whether to increase the number of steps (n_steps) for smoother, more creative transitions.\n\n"
        "Constraints:\n"
        "- Return ONLY JSON, no prose.\n"
        "- JSON keys: focus_attributes (list of short strings), "
        "alpha_start (float), alpha_end (float), n_steps (int or null), explanation (string).\n"
        "- alpha_start should usually be 0.0 or slightly negative if user wants extreme creativity.\n"
        "- alpha_end should be between 0.4 and 1.5.\n"
        "- n_steps should be between 6 and 40 if you choose to override; "
        "set it to null if you want to keep the UI default.\n"
        "- explanation should be 1–3 sentences, plain text."
    )

    user_prompt = f"""
    Here are the two images (rough descriptions only):

    Image 1: {img1_desc}
    Image 2: {img2_desc}

    User's creative intent: "{creative_prompt}"

    Based on this, choose:
    - which key visual attributes or vibes to emphasize when blending,
    - how far to interpolate/extrapolate (alpha_start, alpha_end),
    - whether to override the number of steps (n_steps) for a creative but still coherent sequence.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or "gpt-4.1"
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    raw_content = response.choices[0].message.content

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

    # Clamp alphas into a reasonable range
    if alpha_start < -2.0:
        alpha_start = -2.0
    if alpha_start > 2.0:
        alpha_start = 2.0
    if alpha_end < -2.0:
        alpha_end = -2.0
    if alpha_end > 2.0:
        alpha_end = 2.0

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
