# Chest X-Ray Pneumonia Detection using Deep CNN

## Project Description
This project delivers an end-to-end Computer-Aided Diagnosis (CAD) system designed to automate multi-class pneumonia classification from chest X-ray radiography. Utilizing a custom Deep Convolutional Neural Network (CNN) built with TensorFlow/Keras, the model accurately classifies input images into three distinct clinical categories:
- **NORMAL:** Healthy lung scans with no signs of infection.
- **PNEUMONIA_BACTERIA:** Bacterial pneumonia requiring immediate antibiotic treatment.
- **PNEUMONIA_VIRAL:** Viral pneumonia requiring supportive or antiviral care.

The pipeline integrates advanced data preprocessing, real-time image augmentation, handling of class imbalance via balanced loss weighting (`class_weight='balanced'`), and early stopping regularization.

---

## Dataset
- **Source:** 'https://universe.roboflow.com/inatel/pneumonia-classification-imrcv'(Pneumonia).
- **Structure:** Divided into training, validation, and test sets across 3 category subfolders:
  ```text
  dataset/
  ├── train/
  ├── valid/
  └── test/
 Image Specifications: Grayscale single-channel images, normalized and resized to 256x256 pixels.


---

## Requirements
The project relies on Python 3.8+ and the following core dependencies:
* **tensorflow >= 2.10**
* **numpy**
* **pandas**
* **matplotlib**
* **seaborn**
* **opencv-python**
* **scikit-learn**

All specific versions are pinned inside the `requirements.txt` file.


---


## Model Architecture
The network is a custom multi-layer Convolutional Neural Network (CNN) optimized for $256 \times 256 \times 1$ grayscale inputs:

* **Convolutional Blocks:** Stacked Conv2D layers (3x3 filters) with ReLU activations to extract hierarchical spatial features.

* **Pooling Layers:** MaxPooling2D (2x2) layers to achieve translation invariance and spatial reduction.

* **Regularization:** Dropout layers inserted to prevent feature co-adaptation and combat overfitting.

* **Classification Head:** Flatten layer transitioning into a fully connected Dense layer, ending in a 3-neuron output Dense layer with Softmax activation.

---
## Results
The model was evaluated on an unseen held-out test dataset ($N = 618$ samples):

- **Overall Test Accuracy:** 85.44%
- **Test Loss:** 0.5114
- **Macro Average F1-Score:** 0.84

### Classification Performance Table

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NORMAL** | 0.96 | 0.79 | 0.87 | 231 |
| **PNEUMONIA_BACTERIA** | 0.82 | 0.96 | 0.88 | 240 |
| **PNEUMONIA_VIRAL** | 0.78 | 0.78 | 0.78 | 147 |


> **Key Clinical Takeaway:** The model achieved an exceptional 0.96 Recall for Bacterial Pneumonia (230 out of 240 correctly identified), minimizing critical false negatives.

---

## Prediction Examples
Inference scripts and sample visual outputs can be generated using the saved model (`saved_model/`).

### Example Code Snippet for Inference:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('pneumonia_detection_model.keras')

# Load & preprocess image
img_path = 'new images/sample.jpeg'
img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_names = ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRAL']
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted Class: {predicted_class}")
```
---

## Author
* **Hossam Mohamed Refaat**
