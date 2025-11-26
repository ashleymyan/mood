import logging
import os
from typing import List, Union

import gradio as gr
from PIL import Image
import numpy as np

from vibe_blending import run_vibe_blend_safe, run_vibe_blend_not_safe
from ipadapter_model import create_image_grid

USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "false" #"true"
DEFAULT_CONFIG_PATH = "./config.yaml"

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


def create_gradio_interface():
    theme = gr.themes.Base(
        spacing_size='md', 
        text_size='lg', 
        primary_hue='blue', 
        neutral_hue='slate', 
        secondary_hue='pink'
    )
    
    demo = gr.Blocks(theme=theme)
    with demo:
        gr.Markdown("""
        ## Vibe Blending Demo
        
        This is the demo for the paper "*Vibe Spaces for Creatively Connecting and Expressing Visual Concepts*".
        
        [Paper]() | [Code]() | [Website]()
        
        Given a pair of images, vibe blending will generate a set of images that creatively connect the input images.
        
        **How to use:**
        1. Upload 2 images that share some thematic relationship
        2. Set the start and end α values to control the interpolation weight
        3. (optional) Upload extra images that help to find the commonalities between the input images
        4. (optional) Upload negative images that help to avoid unwanted attributes
        5. Enjoy the creativity!
        """)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        input1 = gr.Image(label="Input 1", show_label=True)
                        input2 = gr.Image(label="Input 2", show_label=True)

                with gr.Group():
                    with gr.Row():
                        alpha_start = gr.Slider(minimum=0, maximum=2, step=0.1, value=0.0, label="Start α", info="interpolation weight")
                        alpha_end = gr.Slider(minimum=0, maximum=2, step=0.1, value=1.0, label="End α", info="use α>1 for extrapolation")
                    # n_steps = gr.Slider(minimum=1, maximum=40, step=1, value=10, label="Number of Output Images")
                    n_steps = gr.Number(value=12, label="Number of Output Images", interactive=True)
                    with gr.Row():
                        extra_images = gr.Gallery(label="Extra Images (optional)", show_label=True, columns=3, rows=2, height=150)
                        negative_images = gr.Gallery(label="Negative Images (optional)", show_label=True, columns=3, rows=2, height=150)
            with gr.Column():
                with gr.Group():
                    # blending_results = gr.Gallery(label="Vibe Blending Results", columns=5, rows=4, height=600)
                    blending_results = gr.Image(label="Vibe Blending Results", show_label=True, height=600)
                    blend_button = gr.Button("🔴 Run Vibe Blending", variant="primary")
        
        # Training wrapper function
        def blend_button_click(input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps):
            input1 = load_gradio_images_helper(input1)
            input2 = load_gradio_images_helper(input2)
            extra_images = load_gradio_images_helper(extra_images)
            negative_images = load_gradio_images_helper(negative_images)

            if extra_images is None:
                extra_images = []
            elif isinstance(extra_images, Image.Image):
                extra_images = [extra_images]

            if negative_images is None:
                negative_images = []
            elif isinstance(negative_images, Image.Image):
                negative_images = [negative_images]

            alpha_weights = np.linspace(alpha_start, alpha_end, n_steps+2)[1:-1].tolist()
            blended_images = run_vibe_blend_not_safe(input1, input2, extra_images, negative_images, DEFAULT_CONFIG_PATH, alpha_weights)
            blended_images = create_image_grid(blended_images, rows=np.ceil(len(blended_images)/4).astype(int), cols=4)
            return blended_images
        
        blend_button.click(blend_button_click, inputs=[input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps], outputs=[blending_results])
        
        example_cases = [
            [Image.open("./images/playviolin_hr.png"), Image.open("./images/playguitar_hr.png"), [], []],
            [Image.open("./images/input_cat.png"), Image.open("./images/input_bread.png"), [], []],
            [Image.open("./images/02140_left.jpg"), Image.open("./images/02140_right.jpg"), [], []],
            #[Image.open("./images/02718_l.jpg"), Image.open("./images/02718_r.jpg"), [], []],
            [Image.open("./images/03969_l.jpg"), Image.open("./images/03969_r.jpg"), [], []],
            [Image.open("./images/04963_l.jpg"), Image.open("./images/04963_r.jpg"), [], []],
            #[Image.open("./images/05358_l.jpg"), Image.open("./images/05358_r.jpg"), [], []],
            [Image.open("./images/00436_l.jpg"), Image.open("./images/00436_r.jpg"), [], []],
            [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg"), [], []],
            [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg"), [Image.open("./images/archi/extra1.jpg"), Image.open("./images/archi/extra2.jpg"), Image.open("./images/archi/extra3.jpg")], []],
            [Image.open("./images/pink_bear1.jpg"), Image.open("./images/black_bear2.jpg"), [], [Image.open("./images/pink_bear1.jpg"), Image.open("./images/black_bear1.jpg")]],
        ]
        gr.Examples(examples=example_cases, inputs=[input1, input2, extra_images, negative_images], outputs=[blending_results])
            
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
