🏦 Loan Approval Prediction System

An end-to-end supervised machine learning project that predicts loan approval based on applicant financial, credit, and asset information. This system demonstrates a complete ML pipeline including data preprocessing, model training, evaluation, model persistence, and deployment through a web application.

📌 Project Overview

Financial institutions need accurate risk assessment before approving loans. This project builds a classification model using historical applicant data to determine whether a loan should be approved or rejected. A Random Forest classifier was selected for its robustness and ability to handle complex feature interactions.

🚀 Key Features

Data cleaning and missing value handling

Categorical feature encoding

Supervised classification model

Model evaluation using Accuracy & Confusion Matrix

Model saving with Pickle for deployment

Streamlit web application for real-time predictions

🛠 Tech Stack
Tool	Purpose
Python	Core programming
Pandas	Data processing
NumPy	Numerical computation
Scikit-learn	ML model training
Streamlit	Web app deployment


📊 Model Performance

Accuracy: ~98%

Low misclassification rate

Strong performance on unseen test data



📂 Project Structure
loan-approval-prediction/
│
├── train_model.py      # Model training & evaluation
├── app.py              # Streamlit web application
├── loan_model.pkl      # Saved trained model
├── data/
│   └── loan_approval_dataset.csv
└── README.md

▶ How to Run
1. Install Dependencies
pip install pandas numpy scikit-learn streamlit

2. Train the Model
python train_model.py

3. Launch Web App
python -m streamlit run app.py

🎯 Conclusion

This project showcases how supervised learning can be applied to real-world financial decision systems, highlighting practical ML development from data preparation to deployment.
