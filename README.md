# **Heart Disease Prediction Using Machine Learning**  

## **Project Overview**  
This project applies machine learning techniques to predict heart disease based on patient data. The dataset consists of 918 records with 12 key features. Various models, including Logistic Regression, Support Vector Machine (SVM), Random Forest, Gradient Boosting, and Multi-Layer Perceptron (MLP), were trained and evaluated. After hyperparameter tuning, **Gradient Boosting** emerged as the best-performing model with **88.6% accuracy and a ROC-AUC of 93.9%**. This project highlights the potential of AI in assisting early heart disease diagnosis.

---

## **Abstract**  
Heart disease is one of the major global health concerns. This project investigates whether machine learning models can accurately predict heart disease using a dataset containing **918 records and 12 features**. Models including **Logistic Regression, SVM, Random Forest, Gradient Boosting, and MLP** were trained and evaluated using **accuracy, F1-score, and ROC-AUC**. **Gradient Boosting** achieved the best performance, with an **88.6% accuracy and 93.9% ROC-AUC**. The study also discusses dataset imbalance, hyperparameter tuning, and real-world deployment challenges.

---

## **Dataset**  
- **Source:** [Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download)
 - **Number of Samples:** 918  
- **Number of Features:** 12  
- **Target Variable:** Heart Disease (0 = No, 1 = Yes)  

### **Key Features:**  
- **Demographics:** Age, Sex  
- **Health Metrics:** Resting Blood Pressure, Cholesterol, Max Heart Rate, Oldpeak  
- **Medical Conditions:** Chest Pain Type, Resting ECG, ST Slope  
- **Exercise Factors:** Fasting Blood Sugar, Exercise-Induced Angina  

---

 **Technologies Used**  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib  
- **Machine Learning Models:** Logistic Regression, SVM, Random Forest, Gradient Boosting, MLP  

---

 **Project Workflow**  
**1️-  Data Preprocessing**  
 missing values has been fixed
 Encoded categorical variables  
 Normalized numerical features where necessary  
 Split data into **80% training and 20% testing**  

 **2️ - Exploratory Data Analysis (EDA)**  
 Older patients with lower max heart rates have a higher risk  
 ST Slope and Chest Pain Type were strong predictors  
 Some cholesterol values were extreme, but retained for analysis  
 **Feature Selection:** Used **Random Forest Feature Importance** to rank key features  

 **3️ - Model Training & Evaluation**  
Five models were tested:  
| Model               | Accuracy | F1-Score | ROC-AUC |  
|---------------------|----------|---------|---------|  
| Logistic Regression | 87.5%    | 89.2%   | 89.6%   |  
| SVM                | 71.7%    | 74.3%   | 79.9%   |  
| Random Forest      | 87.5%    | 88.9%   | 92.3%   |  
| **Gradient Boosting** | **88.6%** | **89.7%** | **93.9%** |  
| MLP                | 83.7%    | 84.5%   | 90.5%   |  

### **4️-  Hyperparameter Tuning**  
 **Gradient Boosting Best Parameters:**  
   - Learning Rate: 0.1  
   - Max Depth: 3  
   - Number of Estimators: 200  
 Optimized using **GridSearchCV & RandomizedSearchCV**  

### **5️ - Model Deployment & Applications**  
 **Clinical Decision Support:** Assists doctors in diagnosis  
 **Wearable Health Devices:** Smartwatches & ECG monitoring  
 **Public Health Analytics:** Identifies high-risk populations for early intervention  

---

## **Challenges & Considerations**  
 **Dataset Imbalance:** More patients with heart disease than without; fixed using stratified sampling  
 **Computational Costs:** Gradient Boosting required high processing power  
 **Ethical Considerations:** AI in healthcare needs transparency & regulatory compliance  

---

 **Future Work & Improvements**  
 **Expand the dataset** with real-world patient data  
 **Fine-tune deep learning models for higher accuracy**  
 **Develop a web app for real-time heart disease risk prediction**  

---

 **How to Run the Project**  
1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/MK-github03/heart_disease_prediction.git  
   ```  
2. **Navigate to the Project Folder**  
   ```bash  
   cd heart-disease-prediction  
   ```  
3. **Create and Activate a Virtual Environment**  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # Mac/Linux  
   venv\Scripts\activate  # Windows  
   ```  
4. **Install Dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  
5. **Run the Jupyter Notebook**  
   ```bash  
   jupyter notebook  
   ```  

---

**License**  
This project is licensed under the MIT License.

