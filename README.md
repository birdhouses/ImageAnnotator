# Image Annotation Application

An image annotation tool that automatically generates captions for images using the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce. This application processes images in a specified folder, generates descriptive captions, formats them into comma-separated keywords using an OpenAI assistant, and saves them as text files.

## Features

- **Automatic Image Captioning**: Generates captions for images using a pre-trained BLIP model.
- **Folder Context in Captions**: Includes the parent folder name in the generated captions to provide additional context.
- **Caption Formatting with OpenAI Assistant**: Formats generated captions into concise, comma-separated keywords using an OpenAI assistant.
- **Batch Processing with Concurrency Control**: Processes all images in a folder with controlled concurrency for optimized performance.
- **Encoding Conversion**: Converts text files to UTF-8 encoding to ensure compatibility.
- **Error Handling**: Robust error handling to continue processing even if some files cause errors.

## Requirements

- **Python 3.7+**
- **PyTorch**
- **Transformers**
- **Pillow**
- **aiofiles**
- **chardet**
- **OpenAI Python Library**
- **python-dotenv**

## Installation

### 1. Clone the Repository

Clone the repository from GitHub and navigate into the project directory.

```bash
git clone https://github.com/birdhouses/ImageAnnotator.git
cd ImageAnnotator
```

### 2. Create a Virtual Environment (Optional)

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Ensure that you have all dependencies installed, including PyTorch configured with CUDA if you have a GPU available.

### 4. Set Up Environment Variables

Create a `.env` file in the root directory to store your environment variables. This file should not be committed to version control. Add your OpenAI API key to the `.env` file.

### 5. Obtain OpenAI API Key

To use the OpenAI assistant feature, you need an OpenAI API key. Sign up for an account at [OpenAI](https://platform.openai.com/), navigate to the **API Keys** section, and generate a new API key.

## Directory Structure

```
├── modules
│   ├── annotator.py         # Contains the annotation logic
│   └── assistant.py         # Contains the Assistant class integrating with OpenAI API
├── app.py                   # Main script to run the annotator
├── images                   # Folder containing images to process
├── .env                     # Environment variables file (not included in version control)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Usage

### 1. Prepare the Images Folder

Place all the images you want to process into a folder, for example, `images` in the project root directory. Supported image formats are `.jpg`, `.jpeg`, and `.png`.

### 2. Run the Application

Execute the main script to process all images in the specified folder, generate captions, format them using the OpenAI assistant, and save them as `.txt` files alongside the images.

```bash
python app.py images
```

Replace `images` with the path to your images folder if different.

### 3. Overwrite Existing Captions (Optional)

By default, the script will not overwrite existing `.txt` files. If you wish to overwrite existing captions, modify the `overwrite` variable in `app.py`:

```python
overwrite = True
```

### 4. Adjusting Concurrency (Optional)

The application uses asynchronous processing with controlled concurrency to optimize performance. The concurrency level can be adjusted in `annotator.py` by modifying the `semaphore` value:

```python
semaphore = asyncio.Semaphore(2)  # Increase the number for higher concurrency
```

## Advanced Usage

### Including Folder Name in Captions

The application includes the parent folder name in the generated captions to provide additional context. For example, if your images are in a folder named `beach`, the captions will include "photo of beach - ..." to provide context.

### Encoding Conversion

If you have existing text files with unknown encoding, you can convert all text files in the folder to UTF-8 encoding using the `convert_to_utf8` function in `annotator.py`.

### Formatting Existing Captions

To format existing captions in `.txt` files, you can call the `process_folder_filewords` function in `annotator.py`.

## Dependencies

Ensure you have the required Python packages installed, including PyTorch, Transformers, Pillow, `aiofiles`, `chardet`, OpenAI, and `python-dotenv`. For optimal performance, especially when processing many images, it's recommended to have a CUDA-compatible GPU.

## Important Notes

- **BLIP Model**: The application uses the `Salesforce/blip-image-captioning-base` model. Ensure you have a stable internet connection to download the model the first time you run the script.
- **OpenAI Assistant**: The assistant is used to format the captions. Ensure that you have set up your OpenAI API key correctly and that your API keys are valid.
- **Performance**: Processing a large number of images can be time-consuming. Using a GPU and adjusting the concurrency level can significantly reduce processing time.

## Troubleshooting

- **CUDA Errors**: If you encounter errors related to CUDA, ensure that your GPU drivers are up to date and that PyTorch is installed with CUDA support.
- **OpenAI API Errors**: Ensure your OpenAI API key is correct and that you have sufficient access rights.
- **File Encoding Issues**: If you experience issues with text file encodings, use the `convert_to_utf8` function to standardize all text files to UTF-8 encoding.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Salesforce Research](https://github.com/salesforce/BLIP) for the BLIP model.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for providing easy access to state-of-the-art models.
- [OpenAI](https://openai.com/) for the assistant capabilities.
- [Pillow](https://python-pillow.org/) for image processing capabilities.
- [aiofiles](https://github.com/Tinche/aiofiles) for asynchronous file operations.

---

**Note**: This application is intended for educational purposes. Please ensure you comply with all relevant laws and terms of service when using pretrained models and processing data.

---

## `.env.example`

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

## Environment Variables

- **OPENAI_API_KEY**: Your OpenAI API key for accessing the assistant functionality.

## Logging

The application uses Python's `logging` module to provide informative messages during execution. By default, the logging level is set to `INFO`. You can adjust the logging level in `app.py` as needed.

```python
logging.basicConfig(level=logging.INFO)
```

## Limitations

- **Assistant Response**: The formatting assistant relies on the OpenAI API response. In cases where the assistant response is not as expected, the original caption may be used.
- **Image Formats**: Only images with `.jpg`, `.jpeg`, and `.png` extensions are processed.
- **Asynchronous Processing**: The application uses asynchronous I/O for reading and writing files, which may not provide performance benefits on systems without I/O bottlenecks.

## Future Enhancements

- **Command-Line Arguments**: Implement command-line arguments to allow users to specify options like overwrite, concurrency level, and folder paths.
- **Progress Indicators**: Add progress bars or status updates to inform users of processing status.
- **Error Reporting**: Improve error messages and provide more detailed troubleshooting steps.