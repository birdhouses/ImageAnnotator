# Image Annotation Application

An image annotation tool that automatically generates captions for images using the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce. This application processes images in a specified folder, generates descriptive captions, formats them into comma-separated keywords using an OpenAI assistant, and saves them as text files.

## Features

- **Automatic Image Captioning**: Generates captions for images using a pre-trained BLIP model.
- **Caption Formatting with OpenAI Assistant**: Formats generated captions into concise, comma-separated keywords using an OpenAI assistant.
- **Batch Processing**: Processes all images in a folder, with optional multithreading for faster execution.
- **Encoding Conversion**: Converts text files to UTF-8 encoding to ensure compatibility.
- **Error Handling**: Robust error handling to continue processing even if some files cause errors.

## Requirements

- **Python 3.7+**
- **PyTorch**
- **Transformers**
- **Pillow**
- **chardet**
- **OpenAI Python Library**
- **dotenv**

## Installation

### 1. Clone the Repository

Clone the repository from GitHub and navigate into the project directory.

### 2. Create a Virtual Environment (Optional)

It's recommended to use a virtual environment to manage dependencies.

### 3. Install Dependencies

Install the required Python packages from the `requirements.txt` file. Ensure that you have all dependencies installed, including PyTorch configured with CUDA if you have a GPU available.

### 4. Set Up Environment Variables

Create a `.env` file in the root directory to store your environment variables. This file should not be committed to version control. Add your OpenAI API key and assistant ID to the `.env` file.

### 5. Obtain OpenAI API Key

To use the OpenAI assistant feature, you need an OpenAI API key. Sign up for an account at OpenAI, navigate to the API Keys section, and generate a new API key.

### 6. Set Up the OpenAI Assistant

The application uses an assistant created in the OpenAI dashboard to format the image captions. Create an assistant, add necessary functions for formatting captions, and save the assistant ID for use in the application.

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

Place all the images you want to process into a folder named `images` in the project root directory. Supported image formats are `.jpg`, `.jpeg`, and `.png`.

### 2. Run the Application

Execute the main script to process all images in the specified folder, generate captions, format them using the OpenAI assistant, and save them as `.txt` files alongside the images.

## Advanced Usage

### Multithreading

If your device can handle it, you can try uncommenting the multithreading feature in `annotator.py` to speed up processing. Adjust the number of worker threads as needed.

### Overwriting Existing Captions

By default, the script will not overwrite existing `.txt` files. You can modify the application to overwrite captions if necessary.

### Encoding Conversion

If you have existing text files with unknown encoding, you can convert all text files in the folder to UTF-8 encoding.

## Dependencies

Ensure you have the required Python packages installed, including PyTorch, Transformers, Pillow, chardet, OpenAI, and python-dotenv. For optimal performance, especially when processing many images, it's recommended to have a CUDA-compatible GPU.

## Important Notes

- **BLIP Model**: The application uses the `Salesforce/blip-image-captioning-base` model. Ensure you have a stable internet connection to download the model the first time you run the script.
- **OpenAI Assistant**: The assistant is used to format the captions. Ensure that you have set up the assistant correctly and that your API keys are valid.
- **Performance**: Processing a large number of images can be time-consuming. Using a GPU and enabling multithreading can significantly reduce processing time.

## Troubleshooting

- **CUDA Errors**: If you encounter errors related to CUDA, ensure that your GPU drivers are up to date and that PyTorch is installed with CUDA support.
- **OpenAI API Errors**: Ensure your OpenAI API key is correct and that you have sufficient access rights.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Salesforce Research](https://github.com/salesforce/BLIP) for the BLIP model.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for providing easy access to state-of-the-art models.
- [OpenAI](https://openai.com/) for the assistant capabilities.
- [Pillow](https://python-pillow.org/) for image processing capabilities.

---

**Note**: This application is intended for educational purposes. Please ensure you comply with all relevant laws and terms of service when using pretrained models and processing data.

---

## `.env.example`

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_ASSISTANT_ID=your_openai_assistant_id
```

Replace `your_openai_api_key` with your actual OpenAI API key and `your_openai_assistant_id` with the ID of your assistant.