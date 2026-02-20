from typing import Dict, Any, List
from io import BytesIO
import base64
import json

from PIL import Image
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
