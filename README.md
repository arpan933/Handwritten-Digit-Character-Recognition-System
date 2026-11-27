# Handwritten Digit and Character Recognition System

## 1. Project overview
This project recognizes a single handwritten digit or English letter from an uploaded image. It performs image preprocessing (grayscale conversion, contrast enhancement, median filtering, resizing and normalization) and uses a Convolutional Neural Network (CNN) trained on the EMNIST ByClass dataset.

## 2. Dataset
- **EMNIST ByClass**: 62 classes (digits 0–9, uppercase A–Z and lowercase a–z).
- The training script downloads the dataset through TensorFlow Datasets the first time it runs. Internet access is required for this initial download.
- Dataset source: Cohen et al., *EMNIST: an extension of MNIST to handwritten letter digits*, 2017. See https://www.nist.gov/itl/products-and-services/emnist-dataset

## 3. Requirements
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended
- Internet for first-time dataset download
- CPU works; GPU can make training faster

## 4. Installation and execution
Open a terminal in this project folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py --epochs 5
streamlit run app.py
```

Training time depends on the computer. For a quick test, use `--epochs 1`; use more epochs for a better model. The model file is saved at `models/emnist_model.keras`.

## 5. How to use
1. Train the model once.
2. Start the application with `streamlit run app.py`.
3. Upload a PNG/JPG image containing one centered handwritten digit or letter.
4. View the predicted class, confidence score and top five predictions.

## 6. Project modules
- Image input module
- Preprocessing module
- CNN classification module
- Result and confidence display module

## 7. Limitations
- Best suited to one character per image, not a full handwritten sentence.
- Similar-looking characters (for example O/0 or I/l/1) may be confused.
- Image quality, rotation, handwriting style and background affect accuracy.
- Confidence is not a guarantee that the prediction is correct.

## 8. Troubleshooting
- If TensorFlow installation fails, use Python 3.10/3.11 in a fresh virtual environment.
- If the app says the model is missing, run `python train.py --epochs 5` first.
- Keep the character centered, high contrast and reasonably large in the uploaded image.

## 9. Academic integrity
Run the program on your own computer, capture genuine output screenshots, and replace the placeholders in the report with your actual training results and screenshots. Do not claim an accuracy value until it is measured on your trained model.
