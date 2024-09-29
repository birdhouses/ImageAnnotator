from modules.annotator import generate_annotations
import sys
import logging
import asyncio

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    folder_name = sys.argv[1]
    overwrite = False

    asyncio.run(generate_annotations(folder_name, overwrite))