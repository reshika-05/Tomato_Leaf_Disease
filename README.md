# 🍅 Tomato Leaf Disease Classification (Flask + Streamlit + PyTorch)

This project is a **Tomato Leaf Disease Detection System** built using:

- **PyTorch** (ResNet18 CNN model)
- **Flask API** (for backend inference)
- **Streamlit UI** (for user-friendly interface)
- **Custom-trained `.pth` model**
- **JSON-based class & remedy mapping**

The system predicts the **Top-3 most likely diseases** from an uploaded tomato leaf image and provides **recommended remedies**.

---

## 🚀 Features

### ✔ Tomato leaf disease prediction  
Uses a fine-tuned **ResNet18** model to classify leaf diseases.

### ✔ Flask REST API  
Accepts an image and returns:
- Top-3 predictions  
- Confidence scores  
- Remedies  

### ✔ Streamlit Web App  
User-friendly UI:
- Upload image  
- View predictions  
- Read remedies  

### ✔ Clean model architecture  
Model loading, transforms, and prediction logic are isolated in `predictor.py`.

### ✔ JSON-driven classes & remedies  
Easy to edit or extend.

---

## 📁 Project Structure

📦 Tomato_Leaf_Disease
├── api.py # Flask API
├── ui.py # Streamlit UI
├── predictor.py # Model + transform + prediction logic
├── train.py # Model training script
├── classes.json # Classes + remedies
├── static/
│ └── index.html # Simple HTML frontend (optional)
├── requirements.txt
├── .gitignore
└── README.md

markdown
Copy code

---

## 🧠 Model Information

- Architecture: **ResNet18**
- Input size: **224 × 224**
- Trained using **CrossEntropyLoss** + **Adam optimizer**
- Dataset classes (from `classes.json`):
  - Target Spot  
  - Mosaic Virus  
  - Yellow Leaf Curl Virus  
  - Bacterial Spot  
  - Early Blight  
  - Healthy  
  - Late Blight  
  - Leaf Mold  
  - Septoria Leaf Spot  
  - Spider Mites  

---

## 🧪 API Usage

### **Endpoint:**
POST /predict

nginx
Copy code

### **Request:**
Send an image file:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@leaf.jpg"
Response:
json

{
  "predictions": [
    {
      "class": "Tomato_Early_blight",
      "confidence": 0.92,
      "remedy": "Use chlorothalonil or copper fungicide; remove lower leaves."
    },
    ...
  ]
}
🌐 Running the Project
1️⃣ Install dependencies
nginx

pip install -r requirements.txt
2️⃣ Start Flask API
nginx

python api.py
Runs on:

cpp

http://127.0.0.1:5000
3️⃣ Start Streamlit UI
arduino
streamlit run ui.py
Opens in browser:

arduino
http://localhost:8501
🎨 HTML Frontend 
Static HTML version located in:

arduino
static/index.html
