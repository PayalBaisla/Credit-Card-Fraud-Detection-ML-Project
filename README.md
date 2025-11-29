# Credit-Card-Fraud-Detection-ML-Project
Machine Learning project using Logistic Regression, StandardScaler, Streamlit UI, Voice Alerts, and Transaction Timeline Graph.


# 💳 Credit Card Fraud Detection System (Machine Learning + Streamlit)

This project detects fraudulent credit card transactions using a Logistic Regression ML model.  
It includes:

✔ Logistic Regression (class_weight="balanced")  
✔ StandardScaler normalization  
✔ Streamlit web application  
✔ Voice alerts using pyttsx3  
✔ Timeline graph using matplotlib  
✔ Transaction history  
✔ Balanced dataset using undersampling  

---

## 📂 Dataset Used

Kaggle Credit Card Fraud Detection Dataset  
➡ https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains **284,807 transactions**, out of which **492 are fraud cases**, making it extremely imbalanced.

---

## 🚀 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Streamlit  
- Matplotlib  
- pyttsx3  
- Jupyter Notebook  

---

## ⚙️ How to Run the Project

### 1️⃣ Install required libraries  

pip install -r requirements.txt
### 2️⃣ Run the Streamlit App  

streamlit run app.py
---

## 📊 Machine Learning Model Used

### **Logistic Regression**
- Simple and efficient for binary classification  
- Works well with large datasets  
- Handles imbalanced classes using `class_weight="balanced"`  
- Fast training and good accuracy  

---

## 📁 Project Structure

project/
│── app.py # Main Streamlit app
│── test.py # Testing script
│── model.pkl # Saved ML model
│── requirements.txt # Libraries required
│── Credit_Card_Fraud_Detection_ML.ipynb
│── Project_Report(ML).docx
│── README.md


---

## 📝 Features of this System

### ✔ Fraud Detection  
Predicts whether a transaction is:
- 🟩 Legitimate  
- 🔺 Fraudulent

### ✔ Voice Output  
Provides voice alerts for results.

### ✔ Timeline Graph  
Shows when and how many times transactions were checked.

### ✔ Transaction History  
Every check is stored with:
- Bank Name  
- Card Number  
- Validity  
- Limit  
- Transaction Amount  
- Timestamp  

---

## 👩‍💻 Developed By  

**Payal Baisla**  
B.Tech CSE  
SDIET College  
Machine Learning Project  
