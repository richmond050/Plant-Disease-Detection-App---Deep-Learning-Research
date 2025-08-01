# 🌿 Plant Disease Detection – Deep Learning Research

This repository contains the complete research phase of a deep learning project for detecting plant diseases from leaf images using the PlantVillage dataset. The work includes data preparation, image preprocessing, model training, evaluation, and insights — focused solely on the experimental notebooks and excluding deployment.

## 🔍 Project Overview

Early and accurate plant disease detection is critical to ensuring crop health and food security. This project applies Convolutional Neural Networks (CNNs) to classify plant leaf images into healthy or diseased categories, using data from the open-source PlantVillage dataset.

The final model was later deployed as a full-stack web app (), but this repository focuses solely on the **deep learning and experimentation phase**.

- **Frontend Repo** for the backend [Link](https://github.com/richmond050/plant-disease-detector-backend)
- **Backend Repo** for the frontend [Link](https://github.com/richmond050/plant-disease-detector-frontend)

## 🧪 Workflow Summary

### 1. **Dataset Setup**
- Used the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (local, not uploaded).
- Dataset organized in `data/raw/PlantVillage/`, with each class stored in a folder.


### 2. **Image Preprocessing**
- Resized images to 224x224 for model input.
- Split into training and validation sets.

### 3. **Model Development**
- Built a Convolutional Neural Network (CNN) using TensorFlow/Keras.
- Trained on 15 classes with ~20k images.


### 4. **Evaluation**
- Achieved high training and validation accuracy (~90%).
- Visualized learning curves (accuracy/loss).
- Saved model as `.keras` format for later deployment.

## 🧠 Model Highlights
- Input shape: `(224, 224, 3)`
- Model type: Custom CNN (no pretrained model used in this phase)
- Trained using GPU (local runtime)


🚫 **Note:** This repo does **not** contain:
- Raw image data
- Trained model files (`.keras` or `.h5`)
- Deployment code (`backend/`, `frontend/` folders are excluded)

## 🚀 Deployment (External)
The trained model was later deployed in a full-stack web app using:
- **Flask** for the backend [Live Site](https://plant-disease-detector-app.netlify.app/)
- **React** for the frontend [Live Site](https://plant-disease-detector-backend-production.up.railway.app/)


## 📌 Future Work
- Addition of more plant species

---

## 🙌 Acknowledgements
- Dataset: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Built with: TensorFlow, Keras, Python, Matplotlib, NumPy, Pandas

---


**Richmond Addai**  
_Data Science | Machine Learning | Software Dev_  
🔗 [LinkedIn](https://www.linkedin.com/in/richmond-addai-6a11a31b1/) 
