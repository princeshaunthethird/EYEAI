# 👁️ EyeAI Vision – Eye Disease Detection System

## 🚀 Overview

EyeAI Vision is an AI-powered web application that detects eye diseases from retinal images using deep learning. The system uses a trained EfficientNet-B0 model to classify images into four categories:

* Cataract
* Diabetic Retinopathy
* Glaucoma
* Normal

The application also provides:

* Prediction confidence
* Class probability distribution
* Grad-CAM visualization (AI explainability)
* Disease description
* Model performance insights

---

## 🧠 Features

* ✅ Deep Learning Model (EfficientNet-B0)
* ✅ Real-time image classification
* ✅ Confidence score output
* ✅ Grad-CAM heatmap visualization
* ✅ Interactive Streamlit UI
* ✅ Probability distribution chart
* ✅ Clean and professional dashboard

---

## 📁 Project Structure

eye_project/
│
├── main/
│   ├── app.py              # Streamlit application
│   ├── train.py            # Model training script
│   ├── best_model.pth      # Trained model file
│
├── requirements.txt        # Dependencies
├── README.txt              # Project documentation

---

## ⚙️ System Requirements

* Python 3.10 or 3.11 (recommended)
* Windows / Linux / macOS
* Minimum 8GB RAM (recommended)

---

## 📦 Installation & Setup

### 1. Extract the Project

Unzip the project folder.

---

### 2. Open Terminal

Navigate to the project directory:

cd path_to_eye_project

---

### 3. Create Virtual Environment

python -m venv venv

---

### 4. Activate Virtual Environment

#### Windows:

venv\Scripts\activate

#### macOS/Linux:

source venv/bin/activate

---

### 5. Install Dependencies

pip install -r requirements.txt

---

### 6. Navigate to Main Folder

cd main

---

### 7. Run the Application

streamlit run app.py

---

### 8. Open in Browser

The app will open automatically or visit:

http://localhost:8501

---

## 🖥️ How the Application Works

### Step 1: Upload Image

* User uploads a retinal image (.jpg / .png)

### Step 2: Preprocessing

* Image is resized to 224x224
* Converted to tensor format

### Step 3: Model Prediction

* Image is passed through EfficientNet-B0
* Output probabilities are computed using Softmax

### Step 4: Result Display

* Predicted class is shown
* Confidence score is displayed

---

## 🔍 Explainability (Grad-CAM)

The system uses Grad-CAM to highlight regions in the image that influenced the model's decision.

* Red areas → High importance
* Blue areas → Low importance

---

## 📊 Output Sections

### 1. Prediction Result

Displays:

* Detected disease
* Confidence percentage

### 2. Probability Breakdown

Bar chart showing probabilities for all classes.

### 3. AI Attention Map

Grad-CAM visualization overlay.

### 4. Model Insights

Displays:

* Validation accuracy
* Model architecture

---

## 🧪 Model Details

* Architecture: EfficientNet-B0
* Framework: PyTorch
* Loss Function: CrossEntropyLoss
* Optimization: Adam Optimizer
* Dataset Split: Train / Validation

---

## ⚠️ Important Notes

* Ensure `best_model.pth` is present inside the `main/` folder.
* Do NOT rename model files.
* Do NOT move files from their original structure.

---

## ❗ Troubleshooting

### Issue: Module not found

Solution:
pip install -r requirements.txt

---

### Issue: cv2 error

Solution:
pip install opencv-python

---

### Issue: Model not loading

Ensure:

* best_model.pth exists
* File path is correct

---

## 📌 Disclaimer

⚠️ This project is for educational purposes only and should not be used for medical diagnosis.

---

## 🏆 Project Highlights

* End-to-end AI pipeline
* Explainable AI integration
* Clean UI/UX design
* Hackathon-ready solution

---

## 👨‍💻 Author

Developed as part of a hackathon project for real-world AI application in healthcare.
