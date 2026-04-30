# Cardiovascular Disease Risk Prediction System

---

## **1. Overview**

The Cardiovascular Disease Risk Prediction System is an end-to-end Machine Learning solution designed to assess the likelihood of heart disease using patient clinical data.

The system integrates:
- Data preprocessing pipeline  
- Multiple supervised learning models  
- Advanced evaluation metrics  
- Explainable AI (SHAP)  
- Real-time user input prediction  
- Automated result storage and reporting  

---

## **2. Problem Statement**

Cardiovascular diseases are a leading cause of mortality worldwide. Early detection plays a critical role in prevention and treatment.

This project predicts whether a patient is at risk of cardiovascular disease based on clinical parameters such as age, blood pressure, cholesterol, and ECG results.

---

## **3. Dataset Information**

Dataset: Heart Disease Dataset (UCI Format)

### **Key Details**
- Total Records: ~1025  
- Number of Features: 13  
- Target Variable:  
  - 0 → No Disease  
  - 1 → Disease  

---

## **4. Feature Description**

| Feature | Description |
|--------|------------|
| age | Age of patient |
| sex | Gender |
| chest_pain_type | Chest pain category |
| resting_blood_pressure | Blood pressure |
| cholestoral | Cholesterol level |
| fasting_blood_sugar | Blood sugar |
| rest_ecg | ECG results |
| Max_heart_rate | Maximum heart rate |
| exercise_induced_angina | Angina presence |
| oldpeak | ST depression |
| slope | ST segment slope |
| vessels_colored_by_flourosopy | Number of vessels |
| thalassemia | Blood disorder type |

---

## **5. Project Structure**


heart-risk-system/
│
├── data/
├── src/
├── models/
├── outputs/
├── main.py
└── requirements.txt


---

## **6. End-to-End Workflow**

### **6.1 Data Loading**
- Load dataset using pandas  
- Validate dataset structure  

---

### **6.2 Exploratory Data Analysis (EDA)**
- Dataset summary  
- Distribution analysis  
- Correlation heatmap  

---

### **6.3 Data Preprocessing**
- One-hot encoding for categorical variables  
- Feature scaling using StandardScaler  

---

### **6.4 Train-Test Split**
- 80% training / 20% testing  
- Stratified sampling  

---

### **6.5 Model Training**

Models used:
- Logistic Regression  
- Random Forest  
- Decision Tree  
- SVM  
- KNN  
- Gradient Boosting  

---

### **6.6 Model Evaluation**

Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

---

## **7. Model Comparison Table**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------|----------|--------|----------|--------|
| Logistic Regression | 0.873 | 0.856 | 0.905 | 0.880 | 0.946 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Decision Tree | 0.985 | 1.000 | 0.971 | 0.986 | 0.986 |
| SVM | 0.951 | 0.936 | 0.971 | 0.953 | 0.976 |
| KNN | 0.834 | 0.845 | 0.829 | 0.836 | 0.957 |
| Gradient Boosting | 0.980 | 0.981 | 0.981 | 0.981 | 0.994 |

---

### **8. Model Comparison Visualization**

![Model Comparison](outputs/plots/model_comparison.png)

---

## **9. ROC Curve**

![ROC Curve](outputs/plots/roc_curve.png)



## **10. Confusion Matrix**

![Confusion Matrix](outputs/plots/confusion_matrix.png)

## **11. Explainable AI (SHAP)**

---

SHAP is used to:
- Interpret model predictions  
- Identify important features  
- Provide transparency in decision-making  

---

## **12. Model Persistence**

- Model saved using `joblib`  
- Scaler stored for consistent predictions  

---

## **13. User Input Prediction**

### **Input**
- Patient clinical parameters  

### **Output**
- Prediction (Disease / No Disease)  
- Probability score  
- Risk category  

---

## **14. Risk Classification**

| Probability | Risk Level |
|------------|-----------|
| < 0.40 | Low |
| 0.40 – 0.70 | Medium |
| > 0.70 | High |

---

## **15. How to Run the Project**

### **Step 1: Clone Repository**

git clone https://github.com/tridevmani/CVD_Mini-project-ML-.git

cd heart-risk-system


---

### **Step 2: Create Virtual Environment**

python -m venv venv
venv\Scripts\activate


---

### **Step 3: Install Dependencies**

pip install -r requirements.txt


---

### **Step 4: Run the Application**

python main.py


---

## **16. GitHub Deployment Steps**


git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/tridevmani/CVD_Mini-project-ML-.git

git push -u origin main


---

## **17. Important Notes**

- Dataset must be placed in `data/heart.csv`  
- Ensure correct Python environment  
- SHAP may take time; reduce sample size if required  

---

## **18. Conclusion**

This project demonstrates a complete Machine Learning pipeline with:

- Data preprocessing and feature engineering  
- Multi-model training and evaluation  
- Advanced performance metrics  
- Explainable AI integration  
- Real-time prediction capability  

It can be extended into a production-ready healthcare decision-support system.
