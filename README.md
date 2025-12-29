# Face Recognition with Transfer Learning (ResNet50)

## Overview  
This project implements a face recognition system using **transfer learning with ResNet50** to classify individual identities from face images.  
The goal of the project is to apply concepts from **biometrics and pattern recognition**—including feature representation, classification, and evaluation—using a deep learning–based approach.

This project was completed as part of **MSU CSE 402: Biometrics and Pattern Recognition**.  
All code in this repository was written by me, and the dataset used is **publicly available**.

The model is trained and evaluated using **10-fold cross validation** to provide a robust estimate of performance.

---

## Key Features  
- Transfer learning using a pretrained **ResNet50** backbone  
- Custom identity classification head  
- 10-fold cross validation with balanced identity splits  
- Image preprocessing and data augmentation  
- GPU/CPU compatible training pipeline  
- Confusion matrix and accuracy-based evaluation  
- Reproducible experiment setup

---

## Model Architecture  
- Backbone: ResNet50 pretrained on ImageNet  
- Input: RGB face images resized to 224×224  
- Output Layer: Fully connected classification layer (number of identities)  
- Loss Function: Cross-entropy loss  

---

## Dataset  
- Publicly available face dataset organized by identity folders
- Kaggle Link to the Dataset: https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
- Each identity represented by multiple face images  
- Dataset is **not included** in this repository  


---

## Training Details  
- Optimizer: Adam  
- Loss Function: Cross-entropy  
- Batch Size: 32  
- Epochs: 10  
- Learning Rate: 1e-4  
- Learning Rate Scheduler: Step decay  
- Data Augmentation:
  - Random resized cropping
  - Horizontal flipping
  - Color jittering  

During training, each fold is trained independently, and accuracy is evaluated on the held-out fold.

---

## Results  
- Fold-level accuracy reported for each cross-validation split  
- Final result reported as **average cross-validation accuracy**  
- Confusion matrix generated across all folds  

Exact metrics and outputs are saved in the `results/` directory.

---

## How to Run  

1. Install dependencies  
pip install -r requirements.txt

2. Run 10-fold cross validation 

---

Course Context  
This project was completed as part of **CSE 402: Biometrics and Pattern Recognition** at **Michigan State University**.  
It applies course concepts such as:
- Biometric feature representation
- Supervised classification
- Cross-validation
- Performance evaluation using confusion matrices
