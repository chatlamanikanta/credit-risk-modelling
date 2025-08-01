import streamlit as st
from predict_helper import predict
st.title("Credit Risk Modelling")

col1,col2,col3 = st.columns(3)
with col1:
    age = st.number_input("Age",min_value=18,max_value=100,step=1)
with col2:
    income = st.number_input("Income",min_value=0,value=1200000)
with col3:
    loan_amount = st.number_input("Loan Amount",min_value=0,value=2560000)

col4,col5,col6 = st.columns(3)
loan_to_income_ratio = loan_amount/income if income>0 else 0
with col4:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")
with col5:
    loan_tenure_months = st.number_input("Loan Tenure (Months)",min_value=0,value=36)
with col6:
    Avg_DPD = st.number_input("Average DPD",min_value=0,value=20)

col7,col8,col9 = st.columns(3)
with col7:
    deliquent_loan_months = st.number_input("Delinquency Ratio",min_value=0,max_value=100,step=1,value=20)
with col8:
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio",min_value=0,max_value=100,step=1,value=30)
with col9:
    number_of_open_accounts = st.number_input("Open Loan Accounts",min_value=0,max_value=4,step=1,value=2)

col10,col11,col12 = st.columns(3)
with col10:
    residence_type = st.selectbox("Residence Type",['Owned', 'Rented', 'Mortgage'])
with col11:
    loan_purpose = st.selectbox("Loan Purpose",['Education', 'Home', 'Auto', 'Personal'])
with col12:
    loan_type = st.selectbox("Loan Type",['Unsecured', 'Secured'])


if st.button("Calculate Risk"):
    probability, credit_score, rating = predict(age,income,loan_amount,loan_tenure_months,
                                                Avg_DPD,deliquent_loan_months,credit_utilization_ratio,
                                                number_of_open_accounts,residence_type,loan_purpose,loan_type)
    st.write(f"Predicted Probability : {probability:.2%}")
    st.write(f"Credit Score : {credit_score}")
    st.write(f"Risk : {rating}")

    input_dict = {
        "Age": age,
        'Income': income,
        'Loan Amount': loan_amount,
        'Loan Tenure (Months)': loan_tenure_months,
        'Average DPD': Avg_DPD,
        'Delinquency Ratio': deliquent_loan_months,
        'Credit Utilization Ratio': credit_utilization_ratio,
        'Open Loan Accounts': number_of_open_accounts,
        'Residence Type': residence_type,
        'Loan Purpose': loan_purpose,
        'Loan Type': loan_type
    }