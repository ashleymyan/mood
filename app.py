import logging
import os
import tempfile
import uuid
from typing import List, Union, Optional
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image
import numpy as np
import pandas as pd

from vibe_blending import run_vibe_blend_safe, run_vibe_blend_not_safe
from ipadapter_model import create_image_grid
from feedback_viewer import create_feedback_viewer_tab, store_feedback_to_hf_dataset
from llm_planner import analyze_pair_with_llm, judge_best_blend, generate_poster_with_text, create_text_overlay_poster

# Hugging Face Datasets for feedback storage
try:
    from datasets import Dataset, load_dataset  # type: ignore
    from huggingface_hub import login  # type: ignore
    HF_DATASETS_AVAILABLE = True
except ImportError:
    Dataset = None  # type: ignore
    load_dataset = None  # type: ignore
    login = None  # type: ignore
    HF_DATASETS_AVAILABLE = False
    logging.warning("Hugging Face datasets not available. Feedback will not be stored.")

USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "false" #"true"
DEFAULT_CONFIG_PATH = "./config.yaml"
# Hugging Face Dataset repository for storing feedback
# Set this to your Hugging Face username/dataset-name, e.g., "your-username/vibe-blending-feedback"
HF_FEEDBACK_DATASET_REPO = os.getenv("HF_FEEDBACK_DATASET_REPO", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)

if USE_HUGGINGFACE_ZEROGPU:
    try:
        import spaces
    except ImportError:
        USE_HUGGINGFACE_ZEROGPU = False
        logging.warning("HuggingFace Spaces not available, running without GPU acceleration")

if USE_HUGGINGFACE_ZEROGPU:
    run_vibe_blend_safe = spaces.GPU(duration=60)(run_vibe_blend_safe)
    run_vibe_blend_not_safe = spaces.GPU(duration=60)(run_vibe_blend_not_safe)

    try:
        from download_models import download_ipadapter
        download_ipadapter()
    except ImportError:
        logging.warning("Could not import download_models")


def create_gif_from_images(images: List[Image.Image], fps: float = 3.0) -> str:
    """
    Create a GIF from a list of PIL Images.
    
    Args:
        images: List of PIL Images to combine into a GIF
        fps: Frames per second for the GIF (default: 3.0)
    
    Returns:
        Path to the temporary GIF file
    """
    if not images:
        return None
    
    # Calculate duration in milliseconds (1000ms / fps)
    duration_ms = int(1000 / fps)
    
    # Create a temporary file for the GIF
    gif_path = os.path.join(tempfile.gettempdir(), f"vibe_blend_{uuid.uuid4().hex}.gif")
    
    # Save as GIF with loop
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0  # 0 = infinite loop
    )
    
    return gif_path


def load_gradio_images_helper(pil_images: Union[List, Image.Image, str]) -> List[Image.Image]:
    """
    Convert various image input formats to a list of PIL Images.
    """
    if pil_images is None:
        return []
    
    # Handle single image
    if isinstance(pil_images, np.ndarray):
        return Image.fromarray(pil_images).convert("RGB")
    if isinstance(pil_images, Image.Image):
        return pil_images.convert("RGB")
    if isinstance(pil_images, str):
        return Image.open(pil_images).convert("RGB")
    
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




def create_vibe_blending_tab():
    """Create a step-by-step vibe blending workflow for filmmakers."""
    with gr.Tab("Poster Lab"):
        with gr.Row(elem_classes=["mosaic-logo-row"]):
            logo_home_button = gr.Button("mosaic", variant="secondary", elem_classes=["mosaic-logo-btn", "compact-btn"])

        with gr.Group(visible=True) as home_page:
            gr.Markdown("""
            <div class="mosaic-hero home-hero">
                <p class="home-tagline">Blend references into poster concept drafts for your film.</p>
            </div>
            """)
            with gr.Row(elem_classes=["home-cta-row"]):
                get_started_button = gr.Button("Get Started", variant="primary", elem_classes=["compact-btn"])

        with gr.Group(visible=False) as step1_page:
            gr.Markdown("## Step 1: Add Reference Images")
            gr.Markdown("Upload two main references. You can also add optional inspiration images and looks to avoid.")
            with gr.Row():
                input1 = gr.Image(label="Reference Image A (main mood)", show_label=True, format="png")
                input2 = gr.Image(label="Reference Image B (secondary mood)", show_label=True, format="png")
            with gr.Row():
                extra_images = gr.Gallery(label="More inspiration images (optional)", show_label=True, columns=3, rows=2, height=180)
                negative_images = gr.Gallery(label="Looks to avoid (optional)", show_label=True, columns=3, rows=2, height=180)
            with gr.Row():
                back_home_button = gr.Button("Back", variant="secondary", elem_classes=["compact-btn"])
                to_step2_button = gr.Button("Continue to Creative Controls", variant="primary", elem_classes=["compact-btn"])

            example_cases = [
                [Image.open("./images/playviolin_hr.png"), Image.open("./images/playguitar_hr.png")],
                [Image.open("./images/input_cat.png"), Image.open("./images/input_bread.png")],
                [Image.open("./images/02140_left.jpg"), Image.open("./images/02140_right.jpg")],
                [Image.open("./images/03969_l.jpg"), Image.open("./images/03969_r.jpg")],
                [Image.open("./images/04963_l.jpg"), Image.open("./images/04963_r.jpg")],
                [Image.open("./images/00436_l.jpg"), Image.open("./images/00436_r.jpg")],
                [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg")],
            ]
            gr.Examples(examples=example_cases, label="Starter reference pairs", inputs=[input1, input2])

            extra_image_examples = [
                [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg"), [Image.open("./images/archi/extra1.jpg"), Image.open("./images/archi/extra2.jpg"), Image.open("./images/archi/extra3.jpg")]],
            ]
            gr.Examples(examples=extra_image_examples, label="Inspiration image examples", inputs=[input1, input2, extra_images])

            negative_image_examples = [
                [Image.open("./images/pink_bear1.jpg"), Image.open("./images/black_bear2.jpg"), [Image.open("./images/pink_bear1.jpg"), Image.open("./images/black_bear1.jpg")]],
            ]
            gr.Examples(examples=negative_image_examples, label="Looks-to-avoid examples", inputs=[input1, input2, negative_images])

        with gr.Group(visible=False) as step2_page:
            gr.Markdown("## Step 2: Creative Controls")
            gr.Markdown("Set your blend range, number of drafts, and creative direction before generation.")
            with gr.Row():
                alpha_start = gr.Slider(minimum=0, maximum=2, step=0.1, value=0.0, label="Blend start", info="Lower values keep Image A stronger at first")
                alpha_end = gr.Slider(minimum=0, maximum=2, step=0.1, value=1.0, label="Blend end", info="Higher values push style further toward Image B")
            n_steps = gr.Number(value=12, label="How many poster drafts to make", interactive=True)
            creative_prompt = gr.Textbox(
                label="Creative direction (optional)",
                placeholder="Example: moody thriller, bold red accents, minimal text area at the top"
            )
            use_llm_judge = gr.Checkbox(
                label="Auto-pick the strongest poster draft",
                value=False,
                info="AI reviews the results and places the strongest draft first"
            )
            judge_criteria = gr.Textbox(
                label="What should matter most? (optional)",
                placeholder="Example: cinematic mood, clear subject, and clean composition",
                visible=False
            )
            with gr.Row():
                back_step1_button = gr.Button("Back", variant="secondary", elem_classes=["compact-btn"])
                generate_button = gr.Button("Generate Poster Drafts", variant="primary", elem_classes=["compact-btn"])

        with gr.Group(visible=False) as step3_page:
            gr.Markdown("## Step 3: Images Being Generated")
            generation_status = gr.Markdown("### Results will appear here after generation starts.")
            blending_results = gr.Gallery(label="Poster Drafts", show_label=True, columns=4, rows=3, interactive=False)
            with gr.Accordion("Full contact sheet", open=False):
                blending_results_grid = gr.Image(label="Poster contact sheet", show_label=True, format="png", interactive=False)
            with gr.Accordion("Quick flip-through", open=False):
                blending_results_gif = gr.Image(label="Poster sequence (GIF)", show_label=True, format="gif", interactive=False)
            llm_explanation = gr.Markdown("### Creative direction notes will appear here")
            judge_result_display = gr.Markdown(visible=False)

            with gr.Row():
                back_step2_button = gr.Button("Back", variant="secondary", elem_classes=["compact-btn"])
                home_from_step3_button = gr.Button("Back to Home", variant="secondary", elem_classes=["compact-btn"])

            with gr.Row():
                to_step4_button = gr.Button("Add Text & Create Poster →", variant="primary", elem_classes=["compact-btn"])

            with gr.Group():
                gr.Markdown("### Share Feedback")
                rating = gr.Radio(label="How useful are these drafts?", choices=["1", "2", "3", "4", "5"])
                feedback_form = gr.TextArea(label="Notes (optional)", show_label=True, lines=1, placeholder="Tell us what worked or what felt off...")
                make_public = gr.Checkbox(label="Share this feedback publicly", value=True, info="Lets others learn from your notes in the feedback viewer")
                feedback_button = gr.Button("Send Feedback", variant="secondary", size="sm", elem_classes=["compact-btn"])

        with gr.Group(visible=False) as step4_page:
            gr.Markdown("## Step 4: Poster Studio")
            gr.Markdown("Pick a draft as your style reference, enter your poster text, and generate a finished poster — both as an AI-created design and as a quick text overlay on your selected draft.")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Click a draft to use it as the poster base**")
                    step4_gallery = gr.Gallery(
                        label="Your Poster Drafts",
                        show_label=False,
                        columns=3,
                        rows=2,
                        height=220,
                        interactive=False,
                    )
                    selected_draft_display = gr.Image(
                        label="Selected Draft (base style)",
                        show_label=True,
                        format="png",
                        interactive=False,
                        height=180,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**Poster Text**")
                    poster_title = gr.Textbox(
                        label="Film Title",
                        placeholder="e.g. INTERSTELLAR",
                    )
                    poster_tagline = gr.Textbox(
                        label="Tagline",
                        placeholder="e.g. Mankind was born on Earth. It was never meant to die here.",
                    )
                    poster_director = gr.Textbox(
                        label="Director",
                        placeholder="e.g. Christopher Nolan",
                    )
                    poster_cast = gr.Textbox(
                        label="Cast",
                        placeholder="e.g. Matthew McConaughey, Anne Hathaway",
                    )
                    poster_year = gr.Textbox(
                        label="Release Year",
                        placeholder="e.g. 2025",
                    )
                    poster_style_notes = gr.Textbox(
                        label="Style notes (optional)",
                        placeholder="e.g. dark dramatic typography, minimal layout, text centered",
                    )

            poster_status = gr.Markdown("### Select a draft above, fill in your poster text, then click **Generate Poster**.")

            with gr.Row():
                back_step3_button = gr.Button("Back", variant="secondary", elem_classes=["compact-btn"])
                home_from_step4_button = gr.Button("Back to Home", variant="secondary", elem_classes=["compact-btn"])
                generate_poster_button = gr.Button("Generate Poster", variant="primary", elem_classes=["compact-btn"])

            with gr.Group(visible=False) as poster_output_group:
                gr.Markdown("### Your Generated Posters")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**AI-Generated Poster** (Ideogram 3.0, new design inspired by your draft's style)")
                        poster_ai_output = gr.Image(
                            label="AI Poster",
                            show_label=False,
                            format="png",
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("**Text Overlay** (your text composited directly onto the selected draft)")
                        poster_overlay_output = gr.Image(
                            label="Text Overlay Poster",
                            show_label=False,
                            format="png",
                            interactive=False,
                        )

        # ── Step 4 state ──────────────────────────────────────────────────────
        selected_draft_state = gr.State(value=None)

        def _page_updates(target: str):
            return (
                gr.update(visible=target == "home"),
                gr.update(visible=target == "step1"),
                gr.update(visible=target == "step2"),
                gr.update(visible=target == "step3"),
                gr.update(visible=target == "step4"),
            )

        def _go_to_step2_if_ready(input1, input2):
            if input1 is None or input2 is None:
                gr.Warning("Please upload both reference images before continuing.")
                return _page_updates("step1")
            return _page_updates("step2")

        def _prepare_generation_view(input1, input2):
            if input1 is None or input2 is None:
                gr.Warning("Please upload both reference images before generating.")
                return (
                    *_page_updates("step2"),
                    gr.update(value="### Add both reference images to continue."),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value="", visible=False),
                )
            return (
                *_page_updates("step3"),
                gr.update(value="### Generating poster drafts..."),
                gr.update(value=None),
                gr.update(value=[]),
                gr.update(value=None),
                gr.update(value="### Creative direction notes will appear here"),
                gr.update(value="", visible=False),
            )

        def _go_to_step4(blending_results):
            """Navigate to Step 4, pre-populating the draft picker gallery."""
            images = []
            if blending_results:
                for item in blending_results:
                    img_path = item[0] if isinstance(item, tuple) else item
                    images.append(img_path)
            return (
                *_page_updates("step4"),
                images,
                None,
                gr.update(value="### Select a draft above, fill in your poster text, then click **Generate Poster**."),
                gr.update(visible=False),
            )

        def _process_input_images(input1, input2, extra_images, negative_images):
            input1 = load_gradio_images_helper(input1)
            input2 = load_gradio_images_helper(input2)
            extra_images = load_gradio_images_helper(extra_images)
            negative_images = load_gradio_images_helper(negative_images)

            if isinstance(input1, list):
                input1 = input1[0] if input1 else None
            if isinstance(input2, list):
                input2 = input2[0] if input2 else None

            if extra_images is None:
                extra_images = []
            elif isinstance(extra_images, Image.Image):
                extra_images = [extra_images]

            if negative_images is None:
                negative_images = []
            elif isinstance(negative_images, Image.Image):
                negative_images = [negative_images]

            return input1, input2, extra_images, negative_images

        def blend_button_click(input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps, creative_prompt, use_llm_judge, judge_criteria):
            input1, input2, extra_images, negative_images = _process_input_images(input1, input2, extra_images, negative_images)

            if input1 is None or input2 is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value="### Creative direction notes will appear here"),
                    gr.update(value="", visible=False),
                    gr.update(value="### Add both reference images before generating."),
                )

            n_steps_eff = max(1, int(n_steps or 12))
            alpha_weights = np.linspace(alpha_start, alpha_end, n_steps_eff + 2)[1:-1].tolist()
            llm_suggestion = analyze_pair_with_llm(input1, input2, creative_prompt)

            # Apply LLM suggestions to blend range / draft count.
            alpha_start_eff = llm_suggestion.get("alpha_start", alpha_start)
            alpha_end_eff = llm_suggestion.get("alpha_end", alpha_end)
            n_steps_eff = max(1, int(llm_suggestion.get("n_steps", None) or n_steps_eff))
            alpha_weights = np.linspace(alpha_start_eff, alpha_end_eff, n_steps_eff + 2)[1:-1].tolist()

            blended_images_list = run_vibe_blend_not_safe(input1, input2, extra_images, negative_images, DEFAULT_CONFIG_PATH, alpha_weights)

            judge_md = ""
            if use_llm_judge:
                judge_result = judge_best_blend(input1, input2, blended_images_list, creative_prompt, judge_criteria)
                best_idx = judge_result["best_index"]
                judge_md = f"**Top pick (Draft #{best_idx}):** {judge_result['reason']}"
                best_image = blended_images_list.pop(best_idx)
                blended_images_list.insert(0, best_image)

            blended_images_grid = create_image_grid(
                blended_images_list,
                rows=np.ceil(len(blended_images_list) / 4).astype(int),
                cols=4,
            )
            explanation_md = f"**What the assistant emphasized:** {', '.join(llm_suggestion.get('focus_attributes', []))}\n\n"
            explanation_md += llm_suggestion.get("explanation", "")
            gif_path = create_gif_from_images(blended_images_list, fps=3.0)
            status_md = f"### Poster drafts are ready. Generated {len(blended_images_list)} concepts."

            return (
                blended_images_grid,
                blended_images_list,
                gif_path,
                explanation_md,
                gr.update(value=judge_md, visible=use_llm_judge),
                status_md,
            )

        # Make judge criteria visible/invisible based on checkbox.
        use_llm_judge.change(
            lambda x: gr.update(visible=x),
            inputs=[use_llm_judge],
            outputs=[judge_criteria]
        )

        def feedback_button_click(rating, feedback_form, make_public, input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps, blending_results):
            """Handle feedback submission and store to Hugging Face Dataset."""
            if not rating:
                gr.Warning("Please select a rating before submitting feedback.")
                return gr.update(value=None), gr.update(value=""), gr.update(value=True)

            if input1 is None:
                gr.Warning("Please upload Reference Image A before submitting feedback.")
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)

            if input2 is None:
                gr.Warning("Please upload Reference Image B before submitting feedback.")
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)

            if blending_results is None or len(blending_results) == 0:
                gr.Warning("Please generate poster drafts first before submitting feedback.")
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)

            input1_img, input2_img, extra_images_processed, negative_images_processed = _process_input_images(
                input1, input2, extra_images, negative_images
            )

            blending_result_images = []
            for item in blending_results:
                image_path = item[0] if isinstance(item, tuple) else item
                if isinstance(image_path, np.ndarray):
                    blending_result_images.append(Image.fromarray(image_path).convert("RGB"))
                elif isinstance(image_path, Image.Image):
                    blending_result_images.append(image_path.convert("RGB"))
                else:
                    blending_result_images.append(Image.open(image_path).convert("RGB"))

            success = store_feedback_to_hf_dataset(
                rating=rating,
                feedback_text=feedback_form or "",
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                n_steps=n_steps,
                input1_image=input1_img,
                input2_image=input2_img,
                extra_images=extra_images_processed if extra_images_processed else None,
                negative_images=negative_images_processed if negative_images_processed else None,
                blending_result_images=blending_result_images,
                is_public=make_public,
            )

            if success:
                gr.Info("Thank you! Your feedback has been submitted successfully.")
                return gr.update(value=None), gr.update(value=""), gr.update(value=True)

            error_msg = "Feedback could not be stored. "
            if not HF_FEEDBACK_DATASET_REPO:
                error_msg += "Please configure `HF_FEEDBACK_DATASET_REPO` environment variable (e.g., 'your-username/vibe-blending-feedback')."
            else:
                error_msg += "Please check the logs for details."
            gr.Warning(error_msg)
            return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)

        get_started_button.click(
            lambda: _page_updates("step1"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        logo_home_button.click(
            lambda: _page_updates("home"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        back_home_button.click(
            lambda: _page_updates("home"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        to_step2_button.click(
            _go_to_step2_if_ready,
            inputs=[input1, input2],
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        back_step1_button.click(
            lambda: _page_updates("step1"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        back_step2_button.click(
            lambda: _page_updates("step2"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        home_from_step3_button.click(
            lambda: _page_updates("home"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )

        generate_event = generate_button.click(
            _prepare_generation_view,
            inputs=[input1, input2],
            outputs=[
                home_page,
                step1_page,
                step2_page,
                step3_page,
                step4_page,
                generation_status,
                blending_results_grid,
                blending_results,
                blending_results_gif,
                llm_explanation,
                judge_result_display,
            ],
        )
        generate_event.then(
            blend_button_click,
            inputs=[input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps, creative_prompt, use_llm_judge, judge_criteria],
            outputs=[blending_results_grid, blending_results, blending_results_gif, llm_explanation, judge_result_display, generation_status],
        )

        feedback_button.click(
            feedback_button_click,
            inputs=[rating, feedback_form, make_public, input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps, blending_results],
            outputs=[rating, feedback_form, make_public]
        )

        # ── Step 3 → Step 4 navigation ─────────────────────────────────────────
        to_step4_button.click(
            _go_to_step4,
            inputs=[blending_results],
            outputs=[
                home_page, step1_page, step2_page, step3_page, step4_page,
                step4_gallery,
                selected_draft_display,
                poster_status,
                poster_output_group,
            ],
        )

        # ── Step 4 back buttons ────────────────────────────────────────────────
        back_step3_button.click(
            lambda: _page_updates("step3"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )
        home_from_step4_button.click(
            lambda: _page_updates("home"),
            outputs=[home_page, step1_page, step2_page, step3_page, step4_page],
        )

        # ── Gallery draft selection ────────────────────────────────────────────
        def _on_draft_selected(evt: gr.SelectData, gallery_images):
            """When a draft thumbnail is clicked, show it as the selected base."""
            if not gallery_images:
                return None, None
            item = gallery_images[evt.index]
            img_path = item[0] if isinstance(item, tuple) else item
            if isinstance(img_path, str):
                pil_img = Image.open(img_path).convert("RGB")
            elif isinstance(img_path, Image.Image):
                pil_img = img_path.convert("RGB")
            elif isinstance(img_path, np.ndarray):
                pil_img = Image.fromarray(img_path).convert("RGB")
            else:
                return None, None
            return pil_img, pil_img

        step4_gallery.select(
            _on_draft_selected,
            inputs=[step4_gallery],
            outputs=[selected_draft_display, selected_draft_state],
        )

        # ── Generate poster ────────────────────────────────────────────────────
        def _generate_poster(
            selected_draft, title, tagline, director, cast, release_year, style_notes
        ):
            if selected_draft is None:
                gr.Warning("Please click a draft above to select a base image.")
                yield (
                    gr.update(value="### Please select a base draft first."),
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(value=None),
                )
                return

            if isinstance(selected_draft, np.ndarray):
                selected_draft = Image.fromarray(selected_draft).convert("RGB")
            elif isinstance(selected_draft, str):
                selected_draft = Image.open(selected_draft).convert("RGB")

            status_md = "### Generating your poster... this may take 20–40 seconds."
            yield (
                gr.update(value=status_md),
                gr.update(visible=False),
                gr.update(value=None),
                gr.update(value=None),
            )

            # PIL overlay (fast, always succeeds)
            overlay_img = create_text_overlay_poster(
                selected_draft,
                title=title,
                tagline=tagline,
                director=director,
                cast=cast,
                release_year=release_year,
            )

            # DALL-E generation (may raise)
            ai_img = None
            try:
                ai_img = generate_poster_with_text(
                    selected_draft,
                    title=title,
                    tagline=tagline,
                    director=director,
                    cast=cast,
                    release_year=release_year,
                    style_notes=style_notes,
                )
                status_done = f"### Poster ready! ({title or 'Untitled'})"
            except Exception as exc:
                status_done = f"### Text overlay ready. AI generation failed: {exc}"

            yield (
                gr.update(value=status_done),
                gr.update(visible=True),
                gr.update(value=ai_img),
                gr.update(value=overlay_img),
            )

        generate_poster_button.click(
            _generate_poster,
            inputs=[
                selected_draft_state,
                poster_title,
                poster_tagline,
                poster_director,
                poster_cast,
                poster_year,
                poster_style_notes,
            ],
            outputs=[poster_status, poster_output_group, poster_ai_output, poster_overlay_output],
        )


# Feedback viewer functions moved to feedback_viewer.py


def create_merged_interface():
    """Create merged interface with both tabs."""
    theme = gr.themes.Base(
        spacing_size='md',
        text_size='lg',
        primary_hue='rose',
        neutral_hue='stone',
        secondary_hue='amber',
    )

    custom_css = """
    :root {
      --mosaic-bg: #f7f2eb;
      --mosaic-surface: #fffdf9;
      --mosaic-ink: #302822;
      --mosaic-accent: #d94841;
      --mosaic-accent-soft: #fce9dc;
      --mosaic-shadow: 0 12px 28px rgba(48, 40, 34, 0.10);
    }

    .gradio-container {
      background:
        radial-gradient(circle at 15% 20%, #fce9dc 0%, rgba(252, 233, 220, 0) 35%),
        radial-gradient(circle at 85% 5%, #f9e3d8 0%, rgba(249, 227, 216, 0) 40%),
        var(--mosaic-bg);
      color: var(--mosaic-ink);
      font-family: "Nunito Sans", "Avenir Next", "Segoe UI", sans-serif;
    }

    .mosaic-hero {
      background: linear-gradient(140deg, #fffdf9 0%, #fff4ea 100%);
      border: 1px solid #f1e0d5;
      border-radius: 20px;
      padding: 20px 22px;
      box-shadow: var(--mosaic-shadow);
      margin-bottom: 10px;
    }

    .mosaic-brand {
      text-transform: lowercase;
      letter-spacing: 0.16em;
      font-weight: 800;
      font-size: 0.88rem;
      color: var(--mosaic-accent);
      margin-bottom: 8px;
    }

    .mosaic-hero h1 {
      margin: 0;
      font-size: 2rem;
      line-height: 1.15;
    }

    .mosaic-hero p {
      margin: 8px 0 0;
      font-size: 1.02rem;
    }

    .mosaic-subnote {
      color: #6b5a4f;
    }

    .home-hero {
      text-align: center;
    }

    .home-tagline {
      margin: 4px 0 !important;
      font-size: 1.28rem !important;
      color: #3b312a;
      line-height: 1.35;
    }

    .home-cta-row {
      justify-content: center !important;
      margin-top: 10px;
    }

    .mosaic-logo-row {
      justify-content: flex-start !important;
      margin-bottom: 8px;
    }

    .mosaic-logo-btn,
    .mosaic-logo-btn button {
      text-transform: lowercase;
      letter-spacing: 0.14em;
      font-weight: 800 !important;
      font-size: 1.5rem !important;
      color: #d94841 !important;
      width: auto !important;
      min-width: 0 !important;
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      padding: 0 !important;
    }

    .mosaic-logo-btn button:hover {
      background: transparent !important;
      color: #c73f38 !important;
    }

    .compact-btn {
      width: fit-content !important;
      min-width: 0 !important;
      padding: 6px 14px !important;
      font-size: 0.92rem !important;
    }

    .gr-group,
    .gr-box {
      background: var(--mosaic-surface) !important;
      border: 1px solid #f0dfd3 !important;
      border-radius: 18px !important;
      box-shadow: var(--mosaic-shadow);
    }

    .gr-button-primary {
      background: linear-gradient(135deg, #d94841, #ef6a4d) !important;
      border: 0 !important;
    }

    .gr-button-secondary {
      border-radius: 999px !important;
    }

    .tab-nav button {
      border-radius: 999px !important;
    }

    .gr-gallery .grid-wrap > div {
      border-radius: 16px !important;
      overflow: hidden !important;
      box-shadow: 0 10px 20px rgba(48, 40, 34, 0.14);
      transition: transform 0.15s ease;
    }

    .gr-gallery .grid-wrap > div:hover {
      transform: translateY(-2px);
    }
    """

    demo = gr.Blocks(theme=theme, css=custom_css)
    with demo:
        # Create both tabs
        create_vibe_blending_tab()
        create_feedback_viewer_tab()
    
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    demo = create_merged_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
