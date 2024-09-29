# modules/annotations.py

import os
import logging
import asyncio
import aiofiles
import io
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import chardet
from modules.assistant import Assistant
from functools import partial

# Initialize BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

async def generate_annotations(folder_name, overwrite=False, max_concurrent_tasks=5):
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

    semaphore = asyncio.Semaphore(2)  # Limit concurrency

    async def semaphore_wrapper(image_path, text_path, filename):
        async with semaphore:
            await generate_annotation_for_image(image_path, text_path, filename, overwrite, parent_folder_name)

    tasks = [
        semaphore_wrapper(image_path, text_path, filename)
        for image_path, text_path, filename in image_text_paths
        if overwrite or not os.path.exists(text_path) or os.path.getsize(text_path) == 0
    ]

    await asyncio.gather(*tasks)
    logging.info("Completed annotation generation for all images.")

async def generate_annotation_for_image(image_path, text_path, filename, overwrite, parent_folder):
    """Generate annotation for a single image and save it to the corresponding .txt file."""
    logging.info(f"Processing image: {filename}")
    try:
        async with aiofiles.open(image_path, 'rb') as image_file:
            image_data = await image_file.read()
            try:
                image = Image.open(io.BytesIO(image_data))
                image.verify()
                image = Image.open(io.BytesIO(image_data))
            except (UnidentifiedImageError, IOError):
                print(f"Skipping non-image or corrupted file: {filename}")
                return
            annotation = await generate_caption_from_image(image, parent_folder=parent_folder)
            logging.info(f"Generated annotation for {filename}: {annotation}")

        if os.path.exists(text_path) and not overwrite:
            async with aiofiles.open(text_path, "r", encoding="utf-8") as file:
                existing_content = await file.read()
            if annotation not in existing_content:
                combined_content = existing_content + "\n" + annotation
            else:
                combined_content = existing_content
                logging.info(f"Annotation for {filename} already exists. Skipping append.")
        else:
            combined_content = annotation

        async with aiofiles.open(text_path, "w", encoding="utf-8") as file:
            await file.write(combined_content)
        logging.info(f"Saved annotation for {filename}")

    except Exception as e:
        logging.error(f"Error generating annotation for {filename}: {e}")

async def generate_caption_from_image(image, parent_folder=None):
    inputs = blip_processor(images=image, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    blip_model.to(device)

    loop = asyncio.get_event_loop()
    blip_generate_partial = partial(blip_generate, blip_model, inputs, 50, 5)
    outputs = await loop.run_in_executor(None, blip_generate_partial)

    output_text = blip_processor.decode(outputs[0], skip_special_tokens=True)

    if parent_folder:
        output_text = f"photo of {parent_folder} - {output_text}"

    formatted = await format_filewords(output_text)

    if 'please provide the' in formatted.lower() or 'sorry' in formatted.lower():
        return output_text
    return formatted

def blip_generate(model, inputs, max_length, num_beams):
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    return outputs

async def format_filewords(filewords):
    question = f"Format this in a comma separated list of words that describe the image. Be concise and clear. {filewords}"
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, Assistant().chat_with_assistant, question, 'skip')
    return f"{answer}"

async def process_folder_filewords(folder_name):
    """Process all filewords in the specified folder using asyncio."""
    async def process_file(file_path):
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                filewords = await file.read()
            if filewords and len(filewords) >= 3:
                formatted_filewords = await format_filewords(filewords)
                if formatted_filewords == 'Please provide the filewords you would like me to format.':
                    formatted_filewords = ''
                if formatted_filewords:
                    async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
                        await file.write(formatted_filewords)
                logging.info(f"Formatted filewords in {os.path.basename(file_path)}")
                await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Error formatting file {os.path.basename(file_path)}: {e}")

    txt_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith(".txt")]
    tasks = [process_file(file_path) for file_path in txt_files]
    await asyncio.gather(*tasks)

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
