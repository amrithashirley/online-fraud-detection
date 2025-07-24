# 🛡️ Online Fraud Payment Detection using Balanced ML Algorithms

This project is a machine learning-based solution to detect fraudulent payment transactions in online systems. It addresses the class imbalance problem using SMOTE and applies various ML models to ensure accurate and reliable fraud prediction.

## 🚀 Features

- Data preprocessing and feature selection
- Handling imbalanced data using SMOTE
- Training multiple machine learning models
- Saving and reusing the best model
- Web interface for user input and fraud prediction
- SQLite database integration (for demo use)

## 📁 Project Structure

- `Dataset/` – contains the raw data files
- `Fraud/` – core project directory (views, URLs, etc.)
- `FraudApp/` – handles business logic and ML integration
- `model/` – contains saved machine learning model files
- `db.sqlite3` – local SQLite database
- `manage.py` – Django project manager
- `requirements.txt` – list of required Python packages
- `runWebServer.bat` – batch file to start the server easily
- `SCREENS.docx` – screenshots of the working project
- `README.md` – this documentation file

## 🧠 Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

> ✅ **Balancing Technique:** SMOTE (Synthetic Minority Oversampling Technique)

## ⚙️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/akhilaankam/online-fraud-detection.git
cd online-fraud-detection
