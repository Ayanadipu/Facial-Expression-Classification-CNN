# Facial Expression Classification using CNN

A deep learning project that classifies facial expressions using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The project also includes a live webcam application that detects faces and predicts their facial expression.

## Project Overview

Facial expression classification is a computer vision task that identifies visible facial expression patterns from images. This project uses a CNN to classify facial images into seven expression categories.

## Features

- Facial expression classification using deep learning
- CNN model trained on the FER-2013 dataset
- Face detection using OpenCV
- Real-time webcam prediction
- Model evaluation using accuracy, loss, and a confusion matrix

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Dataset

**FER-2013 (Facial Expression Recognition 2013)**

The dataset contains grayscale facial images grouped into seven expression classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Model Performance

The trained CNN achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | 54.28% |
| Test Loss | 1.2102 |
| Number of Classes | 7 |

These results are from the current trained model. Performance may vary depending on the input image and lighting conditions.

## Project Structure

```text
Facial-Expression-Classification-CNN/
├── webcam_app.py
├── facial_expression_model.keras
├── class_names.json
├── haarcascade_frontalface_default.xml
├── Facial_Expression_Classification_CA.ipynb
├── requirements.txt
├── README.md
└── screenshots/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ayanadipu/Facial-Expression-Classification-CNN.git
cd Facial-Expression-Classification-CNN
```

### 2. Install dependencies

```bash
pip install tensorflow opencv-contrib-python numpy matplotlib scikit-learn seaborn
```

### 3. Run the webcam application

```bash
python webcam_app.py
```

Allow camera access if prompted. The application will detect a face and display the predicted expression with a confidence score.

Press **Q** to close the webcam window.

## Limitations

- The model's test accuracy is currently 54.28%, so predictions may be incorrect.
- Prediction confidence does not guarantee that the expression is classified correctly.
- Lighting, camera quality, face position, and individual differences can affect results.
- Facial expressions do not reliably reveal a person's actual emotional state.

## Future Improvements

- Improve model accuracy through data augmentation and model tuning.
- Experiment with deeper CNN architectures and transfer learning.
- Improve performance for underrepresented expression classes.
- Evaluate the model on additional facial image datasets.

## Author

**Ayana Dipu**

A deep learning and computer vision project developed for academic purposes.
