# modules/annotations.py

import os
import logging
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import re
import time
import chardet
from modules.assistant import Assistant

# Initialize BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_annotations(folder_name, overwrite=False):
    """Generate annotations for images without corresponding or empty .txt files."""
    logging.info(f"Starting annotation generation for folder: {folder_name}")
    # Collect image and text paths
    image_text_paths = [
        (os.path.join(folder_name, filename),
         os.path.join(folder_name, f"{os.path.splitext(filename)[0]}.txt"),
         filename)
        for filename in os.listdir(folder_name)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    parent_folder_name = Path(folder_name).name

    for image_path, text_path, filename in image_text_paths:
        try:
            if not os.path.exists(text_path):
                generate_annotation_for_image(image_path, text_path, filename, overwrite, parent_folder_name)
                logging.info(f"Annotation generation completed for {filename}.")
        except Exception as e:
            logging.error(f"Error generating annotation for {filename}: {e}")
            continue

    # MAX_WORKERS = 6

    # parent_folder_name = Path(folder_name).name
    # # Use ThreadPoolExecutor for multithreading
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     # Submit tasks for each image
    #     futures = [
    #         executor.submit(generate_annotation_for_image, image_path, text_path, filename, overwrite, parent_folder_name)
    #         for image_path, text_path, filename in image_text_paths
    #         if overwrite or not os.path.exists(text_path) or os.path.getsize(text_path) == 0
    #     ]

    #     # Ensure all futures are completed
    #     for future in as_completed(futures):
    #         try:
    #             future.result()
    #             logging.info("Annotation generation completed for a future.")
    #         except Exception as e:
    #             logging.error(f"Error in multithreaded annotation generation: {e}")

    logging.info("Completed annotation generation for all images.")


def generate_annotation_for_image(image_path, text_path, filename, overwrite, parent_folder):
    """Generate annotation for a single image and save it to the corresponding .txt file."""
    logging.info(f"Processing image: {filename}")
    try:
        # Generate annotation using CLIP or another method
        with open(image_path, 'rb') as image_file:
            try:
                image = Image.open(image_file)
                image.verify()
            except (UnidentifiedImageError, IOError):
                print(f"Skipping non-image or corrupted file")
                return

            annotation = generate_caption_from_image(image, parent_folder=parent_folder)
            logging.info(f"Generated annotation for {filename}: {annotation}")

        # Check if the text file exists and is not empty or overwrite is True
        if os.path.exists(text_path) and not overwrite:
            # Read the existing content
            with open(text_path, "r", encoding="utf-8") as file:
                existing_content = file.read()
            # Check for duplicate annotation
            if annotation not in existing_content:
                # Combine the existing content with the new annotation
                combined_content = existing_content + "\n" + annotation
            else:
                combined_content = existing_content
                logging.info(f"Annotation for {filename} already exists. Skipping append.")
        else:
            # Use only the new annotation
            combined_content = annotation

        # Save the combined content (or just the new annotation)
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(combined_content)
        logging.info(f"Saved annotation for {filename}")

    except Exception as e:
        logging.error(f"Error generating annotation for {filename}: {e}")

def generate_caption_from_image(image, parent_folder=None):
    inputs = blip_processor(images=image, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    blip_model.to(device)

    with torch.no_grad():
        outputs = blip_model.generate(**inputs, max_length=50, num_beams=5)

    output_text = blip_processor.decode(outputs[0], skip_special_tokens=True)

    if parent_folder:
        output_text = f"photo of {parent_folder} - {output_text}"

    formatted = format_filewords(output_text)

    if 'please provide the' in formatted.lower():
        return output_text
    if 'sorry' in formatted.lower():
        return output_text
    return formatted

def format_filewords(filewords):
    assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
    question = f"Format this in a comma separated list of words that describe the image. Be concise and clear. {filewords}"
    answer = Assistant().chat_with_assistant(assistant_id, question, 'skip')
    return f"{answer}"

def convert_to_utf8(folder_name):
    """Convert all text files in the folder to UTF-8 encoding."""
    for filename in os.listdir(folder_name):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_name, filename)
            try:
                with open(file_path, "rb") as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']

                if encoding != 'utf-8':
                    with open(file_path, "r", encoding=encoding, errors='ignore') as file:
                        content = file.read()

                    with open(file_path, "w", encoding='utf-8') as file:
                        file.write(content)
                    logging.info(f"Converted {filename} to UTF-8 encoding.")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")

def process_folder_filewords(folder_name):
    """Process all filewords in the specified folder using parallel processing."""
    def process_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                filewords = file.read()
            if filewords and len(filewords) >= 3:
                formatted_filewords = format_filewords(filewords)
            with open(file_path, "w", encoding="utf-8") as file:
                if formatted_filewords and formatted_filewords == 'Please provide the filewords you would like me to format.':
                    formatted_filewords = ''
                if formatted_filewords:
                    file.write(formatted_filewords)
            logging.info(f"Formatted filewords in {os.path.basename(file_path)}")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Error formatting file {os.path.basename(file_path)}: {e}")

    txt_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith(".txt")]
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, txt_files)
