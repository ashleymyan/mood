import logging
import os
from typing import List, Union

import gradio as gr
from PIL import Image
import numpy as np

from vibe_blending import run_vibe_blend_safe

USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "true"
DEFAULT_CONFIG_PATH = "./config.yaml"

if USE_HUGGINGFACE_ZEROGPU:
    try:
        import spaces
    except ImportError:
        USE_HUGGINGFACE_ZEROGPU = False
        logging.warning("HuggingFace Spaces not available, running without GPU acceleration")

if USE_HUGGINGFACE_ZEROGPU:
    run_vibe_blend_safe = spaces.GPU(duration=60)(run_vibe_blend_safe)

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
    if isinstance(pil_images, Image.Image):
        return [pil_images.convert("RGB")]
    
    if isinstance(pil_images, str):
        return [Image.open(pil_images).convert("RGB")]
    
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
    custom_css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: 0 auto !important;
    }
    """
    
    demo = gr.Blocks(theme=theme, css=custom_css)
    with demo:
        gr.Markdown("""
        ## Vibe Blending Demo
        
        This demo is for the paper "Vibe Spaces for Creatively Connecting and Expressing Visual Concepts".
        
        [Paper]() | [Code]() | [Website]()
        
        Given a pair of images, vibe blending will generate a set of images that creatively connect the input images.
        
        **Instructions:**
        1. Upload 2-10 images that share some thematic relationship
        2. First two images are the images to be blended
        3. The rest are the extra images that help to find the creative connection
        
        """)

        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", show_label=True, columns=3, rows=2, height=400)

            with gr.Column():
                with gr.Accordion("Options", open=True):
                    alpha_start = gr.Slider(minimum=0, maximum=2, step=0.1, value=0, label="Start α (Interpolation Weight)")
                    alpha_end = gr.Slider(minimum=0, maximum=2, step=0.1, value=1, label="End α (Interpolation Weight)", info="use α<1 for interpolation, α>1 for extrapolation")
                    n_steps = gr.Slider(minimum=1, maximum=40, step=1, value=10, label="Number of Output Images")
                    n_clusters = gr.Slider(minimum=5, maximum=50, step=1, value=25, label="Correspondence Matching Clusters")
        with gr.Row():  
            blend_button = gr.Button("🔴 Run Vibe Blending", variant="primary")
        with gr.Row():
            blending_results = gr.Gallery(
                label="Vibe Blending Results", 
                columns=5, 
                rows=4
            )
        
        # Training wrapper function
        def blend_button_click(images, alpha_start, alpha_end, n_steps, n_clusters):
            if not images or len(images) < 2:
                raise gr.Error("Please upload at least 2 images")
            images = load_gradio_images_helper(images)

            alpha_weights = np.linspace(alpha_start, alpha_end, n_steps+2)[1:-1].tolist()
            blended_images = run_vibe_blend_safe(images[0], images[1], images[2:], DEFAULT_CONFIG_PATH, alpha_weights, n_clusters)
            return blended_images
        
        blend_button.click(
            blend_button_click,
            inputs=[input_images, alpha_start, alpha_end, n_steps, n_clusters],
            outputs=[blending_results]
        )

        with gr.Row():
            # Example image sets
            gr.Markdown("## Example Image Sets")
            example_sets = {
                "set 1": ["./images/input_cat.png", "./images/input_bread.png"],
                "set 2": ["./images/archi/input_A.jpg", "./images/archi/input_B.jpg"],
                "set 2 (with extra images)": ["./images/archi/input_A.jpg", "./images/archi/input_B.jpg", "./images/archi/extra1.jpg", "./images/archi/extra2.jpg", "./images/archi/extra3.jpg"],
                "set 3": ["./images/02140_left.jpg", "./images/02140_right.jpg"],
                "set 4": ["./images/02718_l.jpg", "./images/02718_r.jpg"],
                "set 5": ["./images/03969_l.jpg", "./images/03969_r.jpg"],
                "set 6": ["./images/04963_l.jpg", "./images/04963_r.jpg"],
                "set 7": ["./images/05358_l.jpg", "./images/05358_r.jpg"],
                "set 8": ["./images/00436_l.jpg", "./images/00436_r.jpg"],
            }
            
        for set_name, image_paths in example_sets.items():
            with gr.Row():
                with gr.Column(scale=3):
                    add_btn = gr.Button(f"Load {set_name}", size="lg")
                with gr.Column(scale=7):
                    example_gallery = gr.Gallery(
                        value=image_paths,
                        columns=len(image_paths),
                        rows=1,
                        height=300,
                        show_label=False,
                        interactive=False,
                    )
                
                def load_example_set(gallery_images):
                    return [img[0] if isinstance(img, tuple) else img for img in gallery_images]
                
                add_btn.click(
                    load_example_set,
                    inputs=[example_gallery],
                    outputs=[input_images]
            )


    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
