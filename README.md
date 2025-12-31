# Breast Cancer Tumor Classification

This project is a machine learning application that classifies breast tumors as **Benign** or **Malignant**
using a **Logistic Regression** model.  
The model is trained on the Breast Cancer dataset available from `sklearn`.

---

## 🧠 Problem Statement
Breast cancer diagnosis is critical in medical decision-making.  
This project aims to predict whether a tumor is **benign (non-cancerous)** or **malignant (cancerous)** based on
features extracted from a **Fine Needle Aspiration (FNA)** biopsy.

---

## 📊 Dataset Information
- Source: `sklearn.datasets.load_breast_cancer()`
- Type: **Labeled dataset**
- Labels:
  - `0` → Malignant
  - `1` → Benign
- Features are numeric measurements of tumor characteristics.

---

## ⚙️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

---

## 🔍 Model Used
- **Logistic Regression**
- Suitable for **binary classification** problems
- Dataset split:
  - 80% Training
  - 20% Testing

---

## 📈 Model Performance
- Training Accuracy is evaluated using `accuracy_score`
- Test Accuracy is also calculated to check for overfitting

