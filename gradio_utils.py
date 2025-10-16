import copy
import os
import threading
import uuid
import zipfile
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# Constants
TEMP_DIR_BASE = "/tmp/gallery_download"
FILE_DELETION_DELAY = 86400  # 24 hours in seconds
GRID_ROWS = 3
GRID_COLS = 4
IMAGES_PER_GRID = GRID_ROWS * GRID_COLS + 3  # 15 images per grid


def add_download_button(gallery, filename_prefix="output"):
    """
    Add download functionality to a Gradio gallery component.
    
    Args:
        gallery: Gradio gallery component
        filename_prefix (str): Prefix for the downloaded zip file
        
    Returns:
        tuple: (create_file_button, download_button) Gradio components
    """
    
    def make_3x5_plot(images):
        plot_list = []
        
        # Split the list of images into chunks of 15
        chunks = [images[i:i + IMAGES_PER_GRID] for i in range(0, len(images), IMAGES_PER_GRID)]
        
        for chunk in chunks:
            fig, axs = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 9))
            
            # Turn off axes for all subplots
            for ax in axs.flatten():
                ax.axis("off")
                
            # Add images to subplots
            for ax, img in zip(axs.flatten(), chunk):
                img = img.convert("RGB")
                ax.imshow(img)
            
            plt.tight_layout(h_pad=0.5, w_pad=0.3)

            # Generate a unique temporary filename
            filename = uuid.uuid4()
            tmp_path = f"/tmp/{filename}.png"
            
            try:
                # Save the plot to the temporary file
                plt.savefig(tmp_path, bbox_inches='tight', dpi=144)
                
                # Open and process the saved image
                img = Image.open(tmp_path)
                img = img.convert("RGB")
                img = copy.deepcopy(img)
                plot_list.append(img)
                
            finally:
                # Clean up temporary file and matplotlib figure
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                plt.close(fig)
        
        return plot_list
    
    def delete_file_after_delay(file_path, delay):
        """
        Schedule deletion of a file after a specified delay.
        
        Args:
            file_path (str): Path to the file to delete
            delay (int): Delay in seconds before deletion
        """
        def delete_file():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        timer = threading.Timer(delay, delete_file)
        timer.start()
    
    def create_zip_file(images, filename_prefix=filename_prefix):
        """
        Create a zip file containing the selected images.
        
        Args:
            images: List of images from the gallery
            filename_prefix (str): Prefix for the zip filename
            
        Returns:
            gr.update: Gradio update object for the download button
        """
        if images is None or len(images) == 0:
            gr.Warning("No images selected.")
            return None
            
        gr.Info("Creating zip file for download...")
        
        # Extract image paths/objects from gallery format
        images = [image[0] for image in images]
        if isinstance(images[0], str):
            images = [Image.open(image) for image in images]
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{TEMP_DIR_BASE}/{filename_prefix}_{timestamp}.zip"
        os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
        
        # Create zip file with images
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            temp_dir = f"{TEMP_DIR_BASE}/images/{uuid.uuid4()}"
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Save images to the temporary directory and add to zip
                for i, img in enumerate(images):
                    img = img.convert("RGB")
                    img_path = os.path.join(temp_dir, f"image_{i:04d}.jpg")
                    img.save(img_path, quality=95)
                    zipf.write(img_path, f"image_{i:04d}.jpg")
                    
            finally:
                # Clean up the temporary directory
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    os.rmdir(temp_dir)
        
        # Schedule the deletion of the zip file after 24 hours
        delete_file_after_delay(zip_filename, FILE_DELETION_DELAY)
        gr.Info(f"File is ready for download: {os.path.basename(zip_filename)}")
        return gr.update(value=zip_filename, interactive=True)
    
    def warn_on_click(filename):
        """
        Handle download button click and show warning if no file is available.
        
        Args:
            filename: Current filename value from download button
            
        Returns:
            gr.update: Gradio update object for button interactivity
        """
        if filename is None:
            gr.Warning("No file to download, please `📦 Pack` first.")
        interactive = filename is not None
        return gr.update(interactive=interactive)
    
    # Create UI components
    with gr.Row():
        create_file_button = gr.Button(
            "📦 Pack", 
            elem_id="create_file_button", 
            variant='secondary'
        )
        download_button = gr.DownloadButton(
            label="📥 Download", 
            value=None, 
            variant='secondary', 
            elem_id="download_button", 
            interactive=False
        )
        
        # Set up event handlers
        create_file_button.click(
            create_zip_file, 
            inputs=[gallery], 
            outputs=[download_button]
        )
        download_button.click(
            warn_on_click, 
            inputs=[download_button], 
            outputs=[download_button]
        )
    
    return create_file_button, download_button
