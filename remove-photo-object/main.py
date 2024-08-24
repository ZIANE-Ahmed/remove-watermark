import os
import glob
import numpy as np
from PIL import Image
from src.core import process_inpaint
import concurrent.futures
import gc

def process_single_image(file_path, mask_folder, output_folder):
    try:
        # Get the base filename without extension
        base_name = os.path.basename(file_path).split('.')[0]

        # Construct the paths for mask and output
        mask_path = os.path.join(mask_folder, f"{base_name}.png")
        output_path = os.path.join(output_folder, f"{base_name}.jpg")

        # Check if mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return

        # Open the images
        img_input = Image.open(file_path).convert("RGBA")
        result_image = Image.open(mask_path).convert("RGBA")

        # Process the images
        output = process_inpaint(np.array(img_input), np.array(result_image))

        # Save the result
        img_output = Image.fromarray(output).convert("RGB")
        img_output.save(output_path)
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    finally:
        # Clean up to free memory
        gc.collect()

def process_images(input_folder, mask_folder, output_folder, max_workers=4):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all JPG files in the input folder
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, file_path, mask_folder, output_folder): file_path for file_path in image_files}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # To catch and print any exception raised during processing
            except Exception as e:
                file_path = futures[future]
                print(f"Exception raised during processing of {file_path}: {e}")

# Define folders
input_folder = '../images'
mask_folder = '../masks'
output_folder = '../Processed_Images'

# Process all images with up to 4 concurrent threads
process_images(input_folder, mask_folder, output_folder, max_workers=4)
