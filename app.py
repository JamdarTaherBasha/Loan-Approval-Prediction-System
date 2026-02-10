# imprting modules

import streamlit as st

import pickle

import numpy as np

#Load model

model=pickle.load(open("loan_model.pkl","rb"))

st.title("Loan Approval Prediction System")

st.subheader("Enter Applicant Details")

dependents=st.number_input("NUmber of Dependents",0,10)

education=st.selectbox("Education:",["Graduate","Not Graduate"])

self_emp=st.selectbox("Self Employed",["Yes","No"])

income=st.number_input("Annual Income")

loan_amount=st.number_input("Loan Amount")

loan_term=st.number_input("Loan Term(Months)")

cibil=st.number_input("CIBIL Score")

res_assets=st.number_input("Residential Assets Value")

com_assets=st.number_input("commerical Assets Value")

lux_assets=st.number_input("Luxury Assets Value")

bank_assets=st.number_input("Bank Asset Value")

#convert to model format

edu = 1 if education == "Graduate" else 0

self_emp = 1 if self_emp == "Yes" else 0


if st.button("predict Loan Status"):
    features=np.array([[0,dependents,edu,self_emp,income,loan_amount,loan_term,cibil,res_assets,com_assets,lux_assets,bank_assets]])

    prediction=model.predict(features)

    if prediction ==1:
        st.success("Loan Approved")

    else:
        st.error("Loan Rejected")