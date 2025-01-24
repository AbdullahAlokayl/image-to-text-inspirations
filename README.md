# Image Captioning with Fine-Tuned BLIP Model

This project demonstrates how to fine-tune the BLIP (Bootstrapping Language-Image Pre-training) model for generating captions from images. The fine-tuned model is compared against the base model to observe improvements in caption quality. Additionally, the project includes conditional captioning, enabling user-guided caption generation.

## Project Overview

1. **Dataset**:
   - The dataset consists of labeled images stored in a CSV file. Each entry includes an image file path and its corresponding caption.
   - File: `Dataset/metadata.csv`
   - Folder: `Dataset/Train` contains approximately 300 images used for training the model.

2. **Model**:
   - Base Model: `Salesforce/blip-image-captioning-base` from Hugging Face, a state-of-the-art pre-trained model for image-to-text tasks.
   - Fine-tuned Model: Customized on a specific dataset of images and captions to generate high-quality captions that better align with the dataset context.
   - The processor handles tokenizing captions and preparing images for input to the model, ensuring seamless compatibility between the dataset and the BLIP architecture.

3. **Key Features**:
   - **Fine-Tuning**: Improves the base BLIP model's ability to generate context-specific captions by training it on a custom dataset.
   - **Model Comparison**: Outputs from the base model and the fine-tuned model are compared side by side to highlight improvements in caption quality.
   - **Conditional Captioning**: Users can provide an initial text prompt to guide the caption generation process, adding flexibility and control.
   - **Dynamic Visualization**: Captions are displayed over images with proper formatting and wrapping for readability.

## Requirements

### Hardware
- GPU (recommended for faster training and inference)

### Software
Install the required libraries:
```bash
pip install torch torchvision transformers datasets matplotlib
```

## Workflow

### Dataset and Model Files
The dataset and model files can be downloaded from the provided link. Place the following files and folders in the same directory as the notebook:
1. `Dataset` folder (containing the training and test datasets)
2. `model-weights-base-finetuned.pth` (fine-tuned model weights)
3. `processor-config-base-finetuned.json` (processor configuration file)

### 1. Dataset Preparation
Ensure the dataset CSV (`metadata.csv`) is formatted as follows:
```csv
image,text
/full/path/to/image1.jpg,caption for image 1
/full/path/to/image2.jpg,caption for image 2
```
Make sure to replace the image paths with their full paths to avoid file resolution errors. Additionally, ensure that training images are stored in the `Dataset/Train` folder.

### 2. Training
Run the Jupyter Notebook to fine-tune the model:
```
blip-notebook.ipynb
```
Key steps in training:
- **Dataset Loading**: The dataset is loaded from a CSV file, and the image paths are cast into the appropriate format using Hugging Face's `datasets` library.
- **Custom Dataset Class**: A `ImageCaptioningDataset` class preprocesses the images and captions, converting them into tensors for model consumption.
- **Model Setup**: The pre-trained BLIP model is loaded from Hugging Face and prepared for fine-tuning on a GPU.
- **Training Loop**:
  - The model is trained for multiple epochs using the AdamW optimizer with a learning rate of `5e-5`.
  - For each batch of image-caption pairs:
    - The input data is tokenized and processed.
    - The model generates predictions and calculates the loss against the ground truth captions.
    - Gradients are backpropagated, and the model weights are updated.
- **Checkpoint Saving**: The fine-tuned model weights and processor configuration are saved for reuse.

### 3. Inference
Use the fine-tuned model for caption generation and comparison:
1. **Comparison**:
   - Test images are stored in the folder `Dataset/test`.
   - Captions are generated using both the fine-tuned model and the base model.
   - The results are displayed side by side, showcasing the improvements achieved through fine-tuning.

2. **Conditional Captioning**:
   - Users can guide the caption generation process by providing an initial text input.
   - If no input is provided, the model generates captions purely based on the image content.
   - Example:
     - Input text: "An inspiring view of"
     - Output caption: "An inspiring view of the sunset over the mountains."



## File Structure
```
.
├── Dataset
│   ├── metadata.csv          # Training dataset
│   ├── Train                 # Folder containing 300 training images
│   └── test                  # Folder containing test images
├── blip-notebook.ipynb       # Jupyter Notebook for training and inference
├── model-weights-base-finetuned.pth  # Fine-tuned model weights
├── processor-config-base-finetuned.json  # Processor config
└── README.md                 # Project documentation
```

## Results
- The fine-tuned model demonstrates improvements in generating meaningful captions compared to the base model.
- Side-by-side comparisons help visualize the improvements.
- Conditional captioning enables user interaction for customized outputs.

### Visualized Outputs
Below are examples of the model results:
<p align="center">
  <img src="Results\23.png" alt="Example 1" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\7.png" alt="Example 2" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\9.png" alt="Example 3" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\13.png" alt="Example 4" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\14.png" alt="Example 5" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\15.png" alt="Example 6" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\3.png" alt="Example 7" width="500" style="margin-top: 15px; margin-right: 15px;"/>
  <img src="Results\2.png" alt="Example 8" width="500" style="margin-top: 15px; margin-right: 15px;"/>
</p>

## Future Improvements
- Expand the dataset with more diverse images and captions for better generalization.
- Experiment with larger or alternative BLIP architectures.
- Add evaluation metrics like BLEU or CIDEr for quantitative analysis.

## References
- [Hugging Face BLIP Model Documentation](https://huggingface.co/docs/transformers/model_doc/blip)
- [BLIP GitHub Repository](https://github.com/salesforce/BLIP)

## License
This project is open-source and available under the [MIT License](LICENSE).

